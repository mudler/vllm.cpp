# LTX-2.5 — the prompt-side AdaLN path (`use_prompt_adaln_single`)

Row: `LTX25-PROMPT-ADALN`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned; not
edited by this row). Issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644), row 0.

Upstream pins:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `fd4ded7f` |
| huggingface/diffusers | `3a2f35d4` |

Both are read from local checkouts at those revisions, and the golden generator
IMPORTS and EXECUTES the LTX-2 modules rather than restating them.

## 0. What is wrong today

`src/vllm/model_executor/models/ltx2_loader.cpp:988 @ baa92ccf7` sets,
unconditionally:

```cpp
declared.use_prompt_adaln_single = false;
```

and `:573` / `:626` do the same on the two manifest paths. The flag defaults
**TRUE** in both references:

- diffusers `src/diffusers/models/transformers/transformer_ltx2.py:1185` —
  `use_prompt_adaln_single: bool = True`
- LTX-2 `packages/ltx-core/src/ltx_core/model/transformer/model.py:77` — same
  default, and `model_configurator.py:76` / `:138` read it as
  `config.get("use_prompt_adaln_single", True)`

The shipped FP8 DiT carries the 18 tensors the flag builds — **9
`audio_prompt_adaln_single.*` at
`tests/vllm/models/ltx2_fp8_dit_manifest.inc:232-240` and 9
`prompt_adaln_single.*` at `:286-294`**, each stream contributing 6 parameters
(`linear_1`, `linear_2`, `linear`, weight and bias) plus the 3 `F32`
`weight_scale` entries the FP8 file carries beside them. So the flag is TRUE for
the checkpoint this campaign renders. `ltx2.cpp:274-276 @ baa92ccf7` refuses
those tensors by name, so a real render needs `allow_unported_modules=1`
(`src/vllm/multimodal/ltx2_video.cpp:660`) — which reaches the loader lines above
and **silently clears the flag**.

(The split read `12` / `6` until 2026-08-15. It is `9` / `9`: `grep -c` over the
two families in that file returns 9 apiece, and the two cited spans are nine
lines each. The TOTAL of 18 was always right, and so was every conclusion drawn
from it; the breakdown was not. Corrected here because the anchor beside it was
being re-derived anyway, and a citation whose own span contradicts the sentence
that carries it is the defect this repair exists to remove.)

Net effect: every render drops the timestep-conditioned half of the prompt K/V
modulation, keeping only the static `prompt_scale_shift_table`. Nothing observes
it: shapes are unchanged, values stay finite, and the goldens were generated with
`use_prompt_adaln_single=False` (`scripts/gen-ltx2-goldens.py:149`), so the gate
agrees with the defect.

Campaign history this row does **not** re-derive: `ltx-2-5.md` §1.2 already
RETRACTED the "prompt K/V carry no timestep term" claim and recorded that the
shipped checkpoint carries a `[4096, 256]` prompt timestep embedder. What was
never closed is *using* the tensors.

## 1. What upstream does, with anchors

### 1.1 The module

`model.py:222-227` (video) and `:252-257` (audio):

```python
self.prompt_adaln_single = (
    AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
    if self.cross_attention_adaln and self.use_prompt_adaln_single
    else None
)
```

`AdaLayerNormSingle` (`adaln.py:19-45`) is the same brick the port already has
(`Ltx2AdaLayerNormSingle`): `emb.timestep_embedder.linear_1 [dim, 256]`,
`linear_2 [dim, dim]`, `linear [coefficient * dim, dim]`. Coefficient **2** here,
not `adaln_embedding_coefficient()` — shift and scale for the K/V only.

diffusers twin: `transformer_ltx2.py:1255-1259`, `num_mod_params=2`.

Registration order inside `_init_video` puts it between `adaln_single` and
`proj_out`, which is where `EnumerateLtx2DitTensors` already reserved its slot
(the `VT_CHECK` at `ltx2.cpp:274-276 @ baa92ccf7`).

### 1.2 The producer

`transformer_args.py:274-277`, inside `TransformerArgsPreprocessor.prepare`:

```python
prompt_timestep = None
if self.prompt_adaln is not None:
    prompt_timestep, _ = self._prepare_timestep(
        modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
    )
```

Three things this fixes in one line, each of which a shape check cannot see:

1. The input is **`modality.sigma`**, `(B,)` (`modality.py:54`) — the per-sample
   scalar noise level — **not** `modality.timesteps`, which is per-token `(B, T)`.
2. `_prepare_timestep` (`transformer_args.py:173-186`) multiplies by
   `timestep_scale_multiplier` before the embedder, exactly as the port's
   `PrepareTimestep` already does for the main AdaLN.
3. The result is viewed to `(B, -1, 2 * dim)`, i.e. `(B, 1, 2 * dim)` — one row
   broadcast over the prompt tokens.

Wired into both preprocessor kinds at `model.py:313`, `:333` (multimodal) and
`:348`, `:364` (single-modality).

diffusers twin: `transformer_ltx2.py:1536-1547`. diffusers passes `sigma` already
scaled from the pipeline (`pipeline_ltx2_image2video.py:1481` — `sigma=timestep`,
and `timestep` is the scheduler's 0..1000 value), so the two references agree on
the value reaching the embedder; only the place the x1000 happens differs. This
port mirrors LTX-2, so the multiply happens here.

### 1.3 The consumer

`transformer.py:427-447` (`apply_cross_attention_adaln`):

```python
kv_modulation = prompt_scale_shift_table[None, None].to(...)              # :441
if prompt_timestep is not None:                                           # :442
    kv_modulation = kv_modulation + prompt_timestep.reshape(
        batch_size, prompt_timestep.shape[1], 2, -1)                      # :443
shift_kv, scale_kv = kv_modulation.unbind(dim=2)                          # :444
...
encoder_hidden_states = context * (1 + scale_kv) + shift_kv               # :446
```

Reached from `_apply_text_cross_attention` (`:223-251`), which is called for the
video stream at `:288-296` and the audio stream at `:317-325`, passing
`video.prompt_timestep` / `audio.prompt_timestep`. Every block, both streams.

diffusers twin: `transformer_ltx2.py:677-693` (`get_mod_params` over
`prompt_scale_shift_table`), threaded at `:1648-1649`.

Layout consequence: the flat `[B, 1, 2 * dim]` row is read as `[2, dim]` with
**shift first, scale second** — the same order the static table already uses in
`ModulateContext` (`ltx2_dit.cpp:118-129`).

## 2. Scope

**In.**

1. `Ltx2DitParams::use_prompt_adaln_single` is honoured end to end: contract,
   binding, host forward, device forward.
2. The 18 tensors enter `EnumerateLtx2DitTensors` / `BindLtx2DitWeights` when
   `cross_attention_adaln && use_prompt_adaln_single`.
3. `temb_prompt` / `temb_prompt_audio` computed from each stream's own `sigma`
   and threaded into every block's text cross-attention on both the host
   (`ltx2_dit.cpp`) and device (`ltx2_device.cpp`) paths.
4. The three loader `= false` assignments are deleted, and replaced by a guard
   (§3.2) that makes a future silent clear impossible.
5. The `ltx2.cpp:274-276 @ baa92ccf7` refusal is deleted for these two families.
6. Goldens executed from upstream at reduced dims, with a mutation proving the
   new term is load-bearing, and a measured magnitude.

**Out.** Anything the campaign already records as owed: keyframe absolute
position embedding (still genuinely unported, still what
`allow_unported_modules` is for), the caption projections, guidance
perturbations, and the bf16/FP8/NVFP4 stream dtypes on the host forward.

## 3. Design

### 3.1 The seam

`Ltx2DitWeights` gains two `Ltx2AdaLayerNormSingleWeights` members;
`Ltx2BlockArgs` / `BlockArgsDev` gain a `[batch, 1, 2 * width]` prompt-modulation
pointer per stream, `nullptr` when the flag is off. `ModulateContext` and its
device twin take that pointer and add `prompt_mod[b, {0,1} * width + c]` to the
table row before applying `context * (1 + scale) + shift`. `nullptr` gives the
existing static-only behaviour byte-for-byte, which is what keeps every current
golden valid.

The **order of the two additions** is upstream's: the table and the timestep row
are summed FIRST (`:443`), and only then does `(1 + scale)` apply. Folding it the
other way would round differently — the same trap `ProcessOutput`
(`ltx2_dit.cpp:640-642`) already documents.

### 3.2 The prompt-K/V cache, and what replaces the cleared flag

`Ltx2DitForward` already refuses a cache when the flag is on
(`ltx2_dit.cpp:775-780`); with the flag no longer cleared, that refusal becomes
*reachable* rather than dead, and it is correct: the K/V now carry a timestep
term. Nothing in the shipped pipeline passes a cache (`grep` over
`ltx2_pipeline.cpp` finds none), so no caller regresses.

**What replaced the `= false`.** The loader clearing existed so
`EnumerateLtx2DitTensors` would not throw. With the tensors ported the contract
simply includes them, so the assignment has no job left. In its place the loader
asserts the invariant the clearing used to violate:

> the resolved `use_prompt_adaln_single` must equal whether the FILE carries
> `prompt_adaln_single.linear.weight`

A future edit that re-clears the flag then hits a named refusal instead of
quietly dropping 18 tensors. This is deliberately an *equality*, not a one-sided
check: clearing the flag with the tensors present is the defect this row fixes,
and setting it with the tensors absent would bind missing weights.

### 3.3 `allow_unported_modules`

After this row it is scoped to genuinely-unported modules only: the sole flag it
still clears in a config copy is `use_keyframes_abs_pos_embedding`
(`ltx2_loader.cpp:1033-1035`), whose module really is unported. The guard in §3.2
is what makes that scoping structural rather than a comment — the extra `= false`
cannot come back without going red.

`Ltx2AdoptDeclaredDitParams`'s contract-equality check becomes load-bearing in a
second way: a checkpoint whose config declares `use_prompt_adaln_single=false`
while its shapes carry the tensors now produces two DIFFERENT contracts and is
refused, instead of both sides being forced to the same cleared value.

**SUPERSEDED 2026-08-15 by `LTX25-KEYFRAMES-ABS-POS`
([#658](https://github.com/mudler/vllm.cpp/issues/658), landed as `98f8e046d`).**
`keyframes_abs_pos_embedding` is PORTED, so the "sole flag still cleared" above
is now cleared by nothing: `Ltx2AdoptDeclaredDitParams` resolves the declared
flag against what the file carries instead of force-clearing it, and neither
shipped DiT needs `allow_unported_modules` any more. This paragraph is kept as
the state at the time this row ran; the current disposition is
[`ltx25-keyframes-abs-pos.md`](ltx25-keyframes-abs-pos.md) §2. What is unchanged
is the guard in §3.2, which is what makes the scoping structural — and which is
why the keyframes port could retire the clear without re-opening this row's
defect.

## 4. Memory format

Mirrors the existing L2 parity forward exactly: f32 host, f32/bf16 device stream
with the `prompt_scale_shift_table` read at F32 (`ltx2_device.cpp:508-511`). The
prompt modulation is one `[batch, 1, 2 * width]` buffer per stream per forward —
`2 * 4096 * batch` floats for the video stream at full size, computed once
outside the block loop, not per block. No new per-token buffer, so no per-token
byte cost.

## 5. Tests

1. `EnumerateLtx2DitTensors` with the flag ON reproduces upstream
   `named_parameters()` verbatim — names, order, ranks, dims — for a model built
   with `use_prompt_adaln_single=True`. This is the 18-tensor contract.
2. Full dual-stream DiT forward, flag ON, against upstream's executed output.
3. **The mutation**: the same forward with the prompt-AdaLN contribution zeroed
   must go RED against that golden. A term that is present but inert is not a
   port.
4. The flag-OFF goldens stay byte-identical, proving the new path is off when
   upstream's is off.
5. The prompt-K/V cache stays refused with the flag on (existing case).

## 6. Measured magnitude

Recorded in §Outcome, on two fixtures that answer two different questions.

The reduced-dimension generator gives the relative change in the modulated prompt
context and in the DiT's outputs, flag ON vs OFF. That is a GATE FLOOR: a number
below round-off would mean the term is inert and the mutation in §5.3 could not
bite. It is **not** the answer to "does this matter", because both the static
table and the prompt-AdaLN MLP are drawn from the same synthetic init scale, so
every ratio it produces is a property of the fixture.

"Does this matter" is answered on the SHIPPED checkpoint's own weights, run
through upstream's `AdaLayerNormSingle`. That measurement is required before the
row's Outcome may state a magnitude.

## 7. Risks

- **The goldens agree with the defect.** Every existing LTX golden was generated
  with the flag off, so no existing case can fail whatever this row does. The new
  flag-ON case is the only instrument, which is why §5.3 mutates it rather than
  asserting it.
- **`sigma` vs `timesteps`.** Using the per-token `timesteps` instead of the
  per-sample `sigma` produces a same-shaped, finite, wrong result at batch 1 with
  uniform timesteps. The golden runs a batch of 2 with per-token timesteps that
  differ from sigma, so the two are distinguishable.
- **Order within the `[2, dim]` row.** Swapping shift and scale is finite and
  same-shaped. The golden's random weights make it observable.

## 8. Stop conditions

- The flag-ON forward cannot be made to match upstream to the existing
  `kRoundOff` bound: stop and report rather than widening the bound.
- The mutation in §5.3 stays green: the path is not reached, and the row is not
  done. Escalate the mutation's magnitude before concluding anything about
  reachability (issue #604).

## Outcome

### What was measured — (a) the gate floor, on SYNTHETIC weights

The generator emits these into `tests/vllm/models/ltx2_goldens.inc` and prints
them on stderr, from the SAME shared weight stream on both arms (keyed by
parameter name, so every common weight is bit-identical and the difference is the
term and nothing else):

| Quantity | Flag ON vs OFF |
|---|---|
| timestep term vs the static table it is added to (**VIDEO stream**) | `max\|term\|` 0.0252 vs `max\|table\|` 0.0487 — 51.7% |
| block-0 modulated prompt K/V (**VIDEO stream**) | `max\|on-off\|` 0.0310 — 5.82% of `max\|off\|` |
| DiT video output (2 blocks) | 1.4567e-4 — 0.04% of `max\|off\|`, **73x** the gate's 2e-6 floor |
| DiT audio output (2 blocks) | 7.367e-5 — 0.03%, **37x** the floor |

**ALL FOUR ROWS ARE GATE-FLOOR NUMBERS, AND NONE OF THEM ANSWERS "DOES THIS
MATTER".** Corrected 2026-08-13 (issue #644) — this section originally billed the
first two as the answer and disclaimed only the last two as synthetic-bounded.
They have identical provenance: `prompt_scale_shift_table` and every
prompt-AdaLN MLP parameter are drawn from the same `param_spec` rule at
`scale=0.05` (`scripts/gen-ltx2-goldens.py:100-106`), so the ratio between them
is a property of the FIXTURE, not of the conditioning. Vary only the MLP init and
it moves with it: 0.005 → 4.1% / 0.49%, 0.05 (committed) → 51.7% / 5.82%,
0.2 → 1450% / 142%. What these rows are FOR is the mutation below: 73x and 37x
above round-off is what makes a zeroed term detectable.

### What was measured — (b) the SHIPPED checkpoint, which is the answer

Measured 2026-08-13 by loading the real tensors into upstream's own
`AdaLayerNormSingle(inner_dim, embedding_coefficient=2)` (`adaln.py:19-45`, built
by `model.py:223-227` / `:253-257`) and evaluating it on `sigma *
timestep_scale_multiplier` — the file's own config gives 1000 — exactly as
`transformer_args.py:274-278` and `:177` do, then comparing against all 48
`prompt_scale_shift_table` / `audio_prompt_scale_shift_table` tensors it is
summed with at `transformer.py:441-443`.

- File: `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
  (7876 tensors, 1,179,408-byte header; the prompt-AdaLN tensors are BF16 there,
  so they are read directly with no dequantisation step of ours in the path).
- Upstream: `ltx_core` at `fd4ded7f`, imported BY PATH from
  `/home/mudler/_git/LTX-2` with `ltx_core.__file__` asserted under that checkout.
- Sigmas: a uniform grid over the whole range, `linspace(0, 1, 101)`, and
  separately the sampler the file's own scheduler config names —
  `LinearQuadraticScheduler().execute(8)` (`components/schedulers.py:60-88`).

| | video (dim 4096) | audio (dim 2048) |
|---|---|---|
| `rms\|table\|`, 48 blocks | 0.017553 | 0.021925 |
| `rms\|term\|`, uniform σ | 0.236446 | 0.347171 |
| **term/table, RMS** | **1347%** | **1583%** |
| `max\|term\|` / `max\|table\|` | **7119%** | **2817%** |
| term/table RMS, LinearQuadratic 8-step | 1275% | 1492% |

Split by row, on the uniform grid: shift 605% (video) / 461% (audio), scale 1732%
/ 2044%. The scale row is where it concentrates, which is the row that multiplies.

**On the shipped model the timestep term DOMINATES the static table; the table is
the perturbation.** What the pre-row renders applied was
`context * (1 + ~0.018 rms) + ~0.018` where upstream applies
`context * (1 + ~0.32 rms) + ~0.10`. Stated as the quantity actually consumed at
`transformer.py:446`, for a context of unit rms: the modulated context has rms
1.0035 static-only against 1.0915 upstream (video, **+8.8%**) and 1.0033 against
1.3170 (audio, **+31.3%**).

**The synthetic fixture UNDERSTATES the real defect by two orders of magnitude.**
Comparing like with like — the fixture ratio is `max\|term\|` / `max\|table\|`,
and so is the shipped-weights ratio — **per stream, each against its own fixture
denominator**:

| | fixture `max\|term\|`/`max\|table\|` | shipped | understatement |
|---|---|---|---|
| video (`prompt_adaln_single` vs `prompt_scale_shift_table`) | 51.7% | 7119% | **138x** |
| audio (`audio_prompt_adaln_single` vs `audio_prompt_scale_shift_table`) | 40.6% | 2817% | **69x** |

**CORRECTED 2026-08-13 (issue #644): this read "138x and 54x", which is not a
per-stream pair.** The generator emits exactly ONE fixture ratio and it is the
VIDEO stream's (`gen-ltx2-goldens.py`, `measure_prompt_adaln_magnitude`), so 54x
divided the shipped AUDIO number by the VIDEO denominator. It is literally true
about the single published figure and it errs CONSERVATIVE — the real audio
understatement is larger. The audio stream's own fixture ratio, recomputed with
the identical statistic on the identical fixture (the generator is imported, so
the weight stream is bit-identical), is **40.6%** — `max|term|` 0.0201763 vs
`max|table|` 0.0496868 — giving **2817 / 40.6 = 69x**. The video row is
unchanged and was always like-for-like: `max|term|` 0.0252012 vs `max|table|`
0.0487142 = 51.7327%, and 7119 / 51.73 = 137.6.

Reproduce the audio denominator by importing the generator and asking it the same
question the video row is built from (`measure_prompt_adaln_magnitude`), so no
weight is re-drawn:

```python
gen.load_upstream(LTX2); torch.set_grad_enabled(False)
on = gen.build_model("split", False, prompt_adaln=True)
video, audio = gen.build_modalities(False)
scale = float(gen.ARCH["timestep_scale_multiplier"])
amod, _ = on.audio_prompt_adaln_single((audio.sigma * scale).flatten(), hidden_dtype=torch.float32)
amod.abs().max() / on.transformer_blocks[0].audio_prompt_scale_shift_table.abs().max()
```

**Fixed at the source, not only in prose.** The generator emitted both video-only
rows unlabelled, which is how a video denominator came to be applied to an audio
numerator; `gen-ltx2-goldens.py` now names the stream on both, and
`ltx2_goldens.inc` was REGENERATED against `ltx_core` `fd4ded7f` to carry them.
The regeneration diff is those two comment lines and nothing else — every golden
VALUE byte-identical, and the `Regenerate with:` header unchanged because the
committed command was re-run verbatim, so this re-proves provenance as well as
the labels. `test_ltx2.cpp` says the same. No fifth golden was added: the audio
figure is a denominator for this record, not a gate floor, so "ALL FOUR ROWS"
still holds.

Recorded because the original Outcome quoted "roughly half the magnitude" from
the fixture as if it described the checkpoint.

Reproduce with `scripts/measure-ltx2-prompt-adaln.py --ltx2 <LTX-2 checkout>
--checkpoint <the file above>`, committed by this repair so the number is
re-runnable rather than transcribed. It asserts `ltx_core.__file__` under the
named checkout before it reads anything, and nothing of ours is in its numeric
path — the only vllm.cpp input is which tensors to read.

### The mutations

All five run on the committed head, restored byte-for-byte afterwards (source
md5s re-checked). Exit status is the authority; assertion COUNTS are recorded
because doctest's summary and the exit code disagree in both directions.

| # | Mutation | Result |
|---|---|---|
| M1 | host `ModulateContext` ignores `prompt_mod` | RED — 3/35 cases, 6/2435 assertions, exit 1 |
| M2 | device `TextCrossAttentionDev` takes the static-only branch always | RED — 1/15 cases, 6/523 assertions, exit 1 |
| M3 | re-add `use_prompt_adaln_single = false` before the loader guard | RED — the guard throws by name; assertion count DROPS 4826 → 4815, exit 1 |
| M4 | prompt AdaLN driven by `m.timesteps` instead of `m.sigma` | RED — 2/35 cases, 4/2435 assertions, exit 1 |
| M5 | shift and scale rows swapped within the `[2, width]` row | RED — 2/35 cases, 4/2435 assertions, exit 1 |

M3 is the one that cannot be reached by any INPUT, and that is stated rather than
papered over: `ParseLtx2DitParamsFromManifest` derives the flag from the same
manifest the guard reads, so they agree by construction unless an assignment
intervenes — which is exactly the edit the guard exists to catch. The
input-driven half of the same rule lives in `Ltx2AdoptDeclaredDitParams`, where a
config that disagrees with the shapes now produces two different contracts and is
refused; that one is gated by a test with real inputs in both directions.

### The gate

`BUILD_EXIT=0` on every build; build logs grepped for `No space left|BFD assertion`
(0 hits) and `df -h /` logged (88% used, 52G free at the end). Case AND assertion
counts against the `cefacd2d0` baseline, measured by reverting the working tree to
HEAD, rebuilding the four targets and running them, then re-applying the diff and
re-checking its md5 (`03324d42…`, identical before and after):

| Suite | HEAD `cefacd2d0` | this row | delta |
|---|---|---|---|
| `test_ltx2` | 30 cases / 1627 assertions | 35 / 2435 | +5 cases, +808 assertions |
| `test_ltx2_loader` | 24 / 4817 | 26 / 4826 | +2 cases, +9 (new cases minus the assertions the retired unported-family claims took with them) |
| `test_ltx2_device` | 13 / 498 | 15 / 523 | +2 cases, +25 assertions |
| `test_ltx2_video` | 30 / 502 | 30 / 502 | unchanged — see below; 502 is the SKIPPED default |

**What `30 / 502` does and does not say (corrected 2026-08-13).** The
shipped-checkpoint case `ltx2 video: the SHIPPED Lightricks checkpoints parse and
load` is env-gated: with `LTX2_CHECKPOINT_ROOT` unset it prints
`SKIPPED` and returns at `test_ltx2_video.cpp:1620-1625`, so `30 / 502` means the
whole real-header case DID NOT RUN — not "it ran and no assertion counted the
module". (The span starts on the `TEST_CASE` line deliberately. The four gate
lines alone are **not unique** — `ltx2 video: the SHIPPED Lightricks VAEs and
upsampler load` carries a byte-identical `LTX2_CHECKPOINT_ROOT` gate at
`:1757-1760` — so an anchor on the gate alone names two places and identifies
neither. That ambiguity was already there when the anchor read `:1316-1319`;
widening by one line to the case name, which occurs exactly once, is what makes
the citation resolvable.) With the variable pointing at the Lightricks tree the
same binary measures **30 cases / 8734 assertions**, both before and after this repair
(re-measured on this branch, exit 0 in both configurations). Any future quote of
this suite's count owes the configuration alongside it.

**Both figures MOVED, and that is the point of quoting the date with them.**
Re-measured 2026-08-15 on this merge commit, after `LTX25-KEYFRAMES-ABS-POS`
(#658) added cases here: `LTX2_CHECKPOINT_ROOT` unset gives **37 cases / 784
assertions**, set gives **37 / 9031**, exit 0 both ways. The CASE count is the
same in both configurations, so an unchanged case count never distinguishes them.
The 2026-08-13 figures above are kept as what was measured then.

**AND THEY MOVED AGAIN, one merge later — the figures in this paragraph are
SUPERSEDED and are kept only as what was measured on the previous merge.**
`0785cfc4d` (#882) added 306 lines to `tests/vllm/multimodal/test_ltx2_video.cpp`
between `00613767d` and the merge that lands, taking it from **37 to 40**
`TEST_CASE`s (`grep -c '^TEST_CASE'` on both revisions; the same grep returns
matches on `origin/main`, so the count is not a failed pattern). That source
count is what is MEASURED here. The doctest CASE and ASSERTION totals in both
configurations were **NOT re-run** on this merge and are therefore UNKNOWN — not
37 / 784 and 37 / 9031. A number that has moved twice in three days is not one to
carry forward on the argument that it probably did not move a third time, and
this file's own rule is that a count owes its configuration AND its date. Owed
under [#673](https://github.com/mudler/vllm.cpp/issues/673), where this suite's
configuration debt already lives; the `set` arm needs the 18.72 GB NVFP4 and
23 GB FP8 DiTs under `$CHECKPOINT_ROOT`, so no CI host can close it.
`test_ltx2_loader.cpp` and `test_ltx2_device.cpp` are byte-identical across those
two revisions — `git diff --numstat` reports nothing on either, against a
positive control that reports `62 8` on `ltx2_video.cpp` — so their counts below
are unaffected, and `test_ltx2.cpp` differs only by this branch's own edit.

**And CI never sets it — [#673](https://github.com/mudler/vllm.cpp/issues/673),
filed 2026-08-13 as visible debt rather than repaired here.** `grep -rn
CHECKPOINT_ROOT .github/` exits 1 with zero hits while the same pattern matches in
`tests/` and `.agents/` (positive control run in the same command, so this is not
an assertion from a failed grep). CI therefore executes **784 of 9031 assertions —
8.7%** of this suite (measured on the PREVIOUS merge, 2026-08-15; it was 502 of
8734, 5.7%, on 2026-08-13; both SUPERSEDED and un-remeasured after #882, see
above — the RATIO is the finding and it does not depend on the exact totals), at an
identical case count in both configurations, and
`scripts/measure-ltx2-prompt-adaln.py` — which produces every shipped-weights
number in this Outcome — is a manual tool no gate invokes (`grep -rn
measure-ltx2-prompt-adaln` hits only its own usage string and this file). So this
row's checkpoint-derived evidence is **manual and host-local**: reproducible only
on a box carrying the 18.72 GB NVFP4 DiT and the 23 GB FP8 DiT under
`$CHECKPOINT_ROOT`. Wiring checkpoints into CI is explicitly NOT in this row.

The `test_ltx2_video` fixture had to move: it declared a config that omits
`use_prompt_adaln_single` (mirroring the shipped NVFP4 DiT) while its SHAPES said
false, so the config/shape equality check refused it — correctly. It now carries
the module, which is the shipped shape and puts the whole video engine on the new
path.

### What was rejected

- **Widening `modulate`'s kernel contract** with a `rows_per_src_row` divisor, to
  express "one row per batch element broadcast over that element's tokens" in one
  launch. Rejected: it changes a kernel's semantics for a dimension that is 1 or 2
  in every shipped call, and would owe its own red-before evidence. The device
  path loops over batch and offsets the pointers instead.
- **Narrowing the static prompt table to the stream dtype** on the flag-ON path,
  which is literally what upstream's `.to(dtype=x_normed.dtype)` does. Rejected as
  out of scope: the existing static-only path deliberately keeps the table at F32
  (`ltx2_device.cpp`, "a narrowed table would be the dtype rule applied
  backwards"), and changing that polarity is a separate decision. The flag-ON path
  routes the sum through `ada_value`, which stores at the stream dtype — the same
  rounding every other table+modulation sum in that file already has.

### Why the defaults are what they are

`Ltx2DitParams::use_prompt_adaln_single` keeps its `true` default, which is now
honoured rather than overwritten. It matches `model.py:77`,
`model_configurator.py:76`/`:138` and diffusers `transformer_ltx2.py:1185`, and
it matches both shipped DiTs: the FP8 file carries no config at all (so the
default decides), and the NVFP4 file's config OMITS the key — verified by reading
both headers off the NAS. So neither shipped checkpoint is refused by the new
config/shape equality check.

`allow_unported_modules` keeps existing, because
`keyframes_abs_pos_embedding` is genuinely unported and a real render still needs
the opt-in for it. What changed is that it can no longer switch a ported feature
off: the loader asserts the flag against the file instead of clearing it, and
`Ltx2AdoptDeclaredDitParams` clears exactly one flag, for a module nothing
applies. (**SUPERSEDED 2026-08-15**, §3.3: that last module is ported, the opt-in
is needed by neither shipped DiT, and `Ltx2AdoptDeclaredDitParams` clears
nothing.)

### The keyframes claim next door, corrected 2026-08-13

**READ THE SUPERSESSION FIRST.** Everything below is the state on 2026-08-13,
when `keyframes_abs_pos_embedding` was unported and both shipped DiTs were
refused. `LTX25-KEYFRAMES-ABS-POS`
([#658](https://github.com/mudler/vllm.cpp/issues/658), `98f8e046d`) ported the
module on 2026-08-14, retired both refusals and committed
`scripts/measure-ltx2-keyframes-meta.py`, which re-runs the meta-device
observation below on demand instead of quoting it. The FINDING stands and that
port confirms it; the DISPOSITION it argues for — refuse by tensor presence, keep
the opt-in — is retired. Current disposition:
[`ltx25-keyframes-abs-pos.md`](ltx25-keyframes-abs-pos.md) §2.

`ltx2.h` carried, in the same paragraph this row rewrote, *"LTX-2.5's checkpoint
does not carry the parameter"* about `keyframes_abs_pos_embedding`. It is FALSE —
the same class of claim as the `use_prompt_adaln_single=false` assertion this row
exists to remove — and the tree already contradicted it twice
(`.agents/model-matrix.md`, `tests/vllm/multimodal/test_ltx2_video.cpp:1608-1609`).
Read straight off both files' headers, and run through upstream's own loader and
configurator:

| | FP8 (`vonkaiser`) | NVFP4 (first-party) |
|---|---|---|
| carries `keyframes_abs_pos_embedding` | YES — `F8_E4M3 [1, 4096]` + F32 scale | NO |
| declares the flag in `__metadata__` | **no `__metadata__` AT ALL** | `true` |
| `LTXModelConfigurator.from_metadata` | **RAISES** `KeyError: 'caption_channels'` | builds it, `[1, 4096]` |

So the two files each contradict one half of the retired claim, and neither
supports it. Two corrections to the reasoning that came with the finding, both
measured rather than read:

- The FP8 file does not "resolve the flag `False` at `model_configurator.py:82`".
  Upstream never reaches line 82 on it: `_build_caption_projections` indexes
  `caption_channels` on the empty config first and raises. That file ships no
  config, so what its flag resolves to is decided entirely out of band — and the
  tensor it carries is trained (`.agents/specs/ltx-2-5.md` §3.1 reads its bytes).
- On the NVFP4 file the flag IS on and the module IS built, but the tensor is
  absent — 0 of its 7876 entries match `keyframes_abs_pos`.

  **CORRECTED AGAIN 2026-08-13, same issue.** This bullet said the parameter
  "keeps `torch.zeros(1, inner_dim)` (`model.py:217-219`) through
  `load_state_dict(..., strict=False)` (`loader/single_gpu_model_builder.py:98`)
  — a genuine no-op there". **It is not a no-op.** The claim quoted that line
  while dropping the `assign=True` that is ON THE SAME LINE, and upstream builds
  on the **meta device** — `loader/helpers.py:84-95`, `create_meta_model`:
  `with torch.device("meta"): configurator.from_metadata(...)` at `:90-91`. A key
  absent from the state dict is therefore never materialised at all; it is not a
  zero, it is an **unmaterialised meta parameter**.

  Run 2026-08-13 through upstream's own `create_meta_model` on this file's real
  `__metadata__` (read with upstream's own `read_model_metadata` /
  `SafetensorsModelStateDictLoader`, which JSON-decodes each value —
  `sft_loader.py:58-74`; the flag lives at `config.transformer`, not at the top
  level):

  ```text
  config.transformer['use_keyframes_abs_pos_embedding'] = True
  keyframes_abs_pos_embedding: shape=(1, 4096) dtype=torch.float32 device=meta is_meta=True
  supports_keyframes_abs_pos_embedding (BEFORE load) : False
  after load_state_dict(sd, strict=False, assign=True):
    neighbour patchify_proj.weight : device=cpu is_meta=False   <- materialised
    keyframes_abs_pos_embedding    : device=meta is_meta=True
    in missing_keys                : True
    reading the value RAISES       : RuntimeError: Tensor.item() cannot be called on meta tensors
  supports_keyframes_abs_pos_embedding (AFTER load)  : False
  ```

  **Upstream says exactly this itself**, and then never asks.
  `supports_keyframes_abs_pos_embedding` (`model.py:166-173`) returns `False` for
  "a model whose config set the flag but whose checkpoint carried no weight for it
  (the parameter would still be on `meta`)", and
  `enable_keyframes_abs_pos_embedding` (`model.py:175-200`) exists because such a
  parameter "would fail at the first forward". **Both are defined and never
  called** — one `grep -rn` hit each across the whole `Lightricks/LTX-2` checkout
  at `fd4ded7f`, the definition itself, re-confirmed here rather than transcribed.

  Polarity is not what fails. `apply_keyframes_absolute_embedding` is
  `hidden_states + mask * embedding` (`transformer_args.py:23-43`, the sum at
  `:43`), so **real** zeros would be inert — the mechanism claim fails on `meta`,
  not on additivity.

  This **strengthens** the row's conclusion and changed nothing downstream on
  2026-08-13: the refusal stayed keyed on tensor presence for the FP8 file, and
  the NVFP4 file was refused by flag in `ParseLtx2DitParams` — where upstream, had
  it loaded, would carry a parameter its own guard reports as unsupported. Both
  refusals are GONE as of `98f8e046d` (#658), and the meta observation above is
  what let that row load the NVFP4 file and apply nothing rather than refuse it or
  synthesise a zero. It is now re-runnable rather than quoted:
  `scripts/measure-ltx2-keyframes-meta.py`, committed by #658 and RE-RUN on this
  merge commit against both shipped files, exit 0. It reproduces every line of
  the transcript above and carries its own positive control — a neighbour
  (`scale_shift_table`) that materialises `device=cpu is_meta=False` on the same
  load, so `is_meta=True` on `keyframes_abs_pos_embedding` is the parameter's
  state and not a loader that never ran. It also settles the FP8 half by
  execution: 2 of 6124 keys match, `[1, 4096]` `F8_E4M3` with 4096 of 4096 bytes
  NON-ZERO plus a rank-0 `F32` scale, against `__metadata__ keys : NONE`.

It is also not a keyframe-only feature: `transformer_args.py:269` applies it on
every `prepare` whose `keyframes_mask` is set, and `tools.py:186-196` sets that
mask unconditionally on the target's first latent frame. (Diffusers' own pipeline
does not consume it — `.agents/specs/ltx-2-5.md` §3.1 records that — but `ltx_core`
is what this campaign ports, and `ltx_core` does.)

**The refusal keying did NOT change here, and that was the decision, not an
omission.** `ltx2_loader.cpp`'s `RefuseUnported` fired on the TENSORS the file
carries. Keying it on the resolved flag instead would, on the FP8 DiT, read a
DEFAULT rather than the file — because that file declares nothing — and would
therefore have loaded it silently while discarding a trained `[1, 4096]`
parameter. Tensor presence was the only signal that file actually carries, and
refusing loudly with an opt-in was strictly safer than resolving quietly. No
behaviour changed in this row, so no new gate was owed; the refusal MESSAGE
changed, because it asserted the implication that is false in both directions.

**RETIRED 2026-08-14 by #658.** Both refusals are gone, because the module is
ported: nothing is keyed on the tensors and nothing is keyed on the flag, and the
declared flag is now RESOLVED against what the file carries
(`Ltx2AdoptDeclaredDitParams`). The reasoning above is why the port had to settle
the NVFP4 arm by EXECUTION before it could retire either refusal.

### The claims repair's own gate (2026-08-13)

Nothing executable changed except three refusal MESSAGES, so the numbers are
expected to be identical and the point is that they are:

- `BUILD_EXIT=0`; build logs grepped for `No space left|BFD assertion` — 0 hits;
  `df -h /` 92% used, 37G free at the end.
- `ctest -N` = **423**. Full `ctest -j8` = 422/423 with `test_serve_low_tools`
  starved under `-j` (a known parallel flake); serially **1/1 PASS, exit 0**.
  2 skipped (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`) as
  on the baseline.
- Suite counts, unchanged from the row above: `test_ltx2` 35/2435,
  `test_ltx2_loader` 26/4826, `test_ltx2_device` 15/523, `test_ltx2_video`
  30/502 skipped-default and **30/8734** with `LTX2_CHECKPOINT_ROOT` set. Exit 0
  on all five runs.
- `tests/vllm/models/ltx2_goldens.inc` REGENERATED from
  `scripts/gen-ltx2-goldens.py` against `ltx_core` `fd4ded7f`: every golden VALUE
  byte-identical, the diff is the comment block alone. That re-proves provenance
  as well as the wording.

An earlier full run was voided rather than reported: another session's
disk-pressure cleanup deleted `build/` while ctest was at 421/423, and the last
two tests recorded `Not Run — Failed to change working directory`. A run whose
tree vanished under it is not a result; it was rebuilt and re-run from scratch.

### Two divergences from upstream, recorded rather than fixed

- **We are stricter than upstream about a config that disagrees with its file.**
  Upstream loads with `load_state_dict(..., strict=False, assign=True)`
  (`loader/single_gpu_model_builder.py:98` — quoted in full, because dropping the
  `assign=True` is exactly what made the keyframes claim above wrong), so a config
  declaring
  `use_prompt_adaln_single=false` over a file that carries the module would build
  no module, drop 18 tensors on the floor and run flag-OFF without a word. §3.2's
  equality check refuses that. Ours is better; it is still a DIVERGENCE, not a
  mirror, and it is named here so it is not later mistaken for ported behaviour.
- **We refuse `keyframes_abs_pos_embedding` by tensor presence** where upstream
  would take an out-of-band config's word for it (above). Same shape of
  divergence, same reason it stood. **RETIRED 2026-08-14 by #658**: the module is
  ported, nothing refuses it, and the declared flag is resolved against the file
  the way upstream's own `supports_keyframes_abs_pos_embedding` resolves it. Only
  the first divergence survives.

### The reconciliation onto `main`, and its gate (2026-08-15)

The branch sat 120 commits behind and conflicted on ten files. Its port half had
already landed as `65e79eee5` (#654), and `98f8e046d` (#658) then ported
`keyframes_abs_pos_embedding`, which is why four of the five SOURCE conflicts were
this branch re-asserting retired refusals. All five took `main`'s side;
`tests/vllm/models/ltx2_goldens.inc` was REGENERATED with the committed command
against `ltx_core` `fd4ded7f` (`REGEN_EXIT=0`) rather than hand-merged, and
differs from `main` by 12 insertions and 7 deletions, all comment lines: every
golden VALUE on both sides byte-identical.

Every published number was re-derived on the merged tree rather than carried
forward, and none moved: the shipped-weights table above via
`scripts/measure-ltx2-prompt-adaln.py` (exit 0), and the audio fixture
denominator 40.6068% by importing the generator so no weight is re-drawn.

Suite counts on the merged tree, exit 0 each, against the 2026-08-13 figures:

| Suite | 2026-08-13 | 2026-08-15 (this merge) |
|---|---|---|
| `test_ltx2` | 35 / 2435 | 43 / 4388 |
| `test_ltx2_loader` | 26 / 4826 | 28 / 4978 |
| `test_ltx2_device` | 15 / 523 | 18 / 546 |
| `test_ltx2_video` | 30 / 502 unset, 30 / 8734 set | 37 / 784 unset, 37 / 9031 set — **SUPERSEDED, not re-run after #882 took the file to 40 `TEST_CASE`s** |

The growth is #658's, not this reconciliation's: no `src/` or `include/` file
differs from `origin/main` on this branch.

All five mutations re-run on this tree, each BUILT, each restored byte-for-byte:

| # | Mutation | BUILT | Result |
|---|---|---|---|
| M1 | host `ModulateContext` ignores `prompt_mod` | YES | RED — `test_ltx2` 3/43 cases, 6/4388 assertions, exit 1 |
| M2 | device path takes the static-only branch always | YES | RED — `test_ltx2_device` 1/18, 6/546, exit 1 |
| M3 | re-add `use_prompt_adaln_single = false` before the guard | YES | RED — `test_ltx2_loader` 1/28 cases, exit 1, and `test_ltx2_video` 29/37, exit 1 |
| M4 | prompt AdaLN driven by `m.timesteps` instead of `m.sigma` | YES | RED — `test_ltx2` 2/43, 4/4388, exit 1 |
| M5 | shift and scale rows swapped in the `[2, width]` row | YES | RED — `test_ltx2` 2/43, 4/4388, exit 1 |

**M5's first attempt was a silent NO-OP and is recorded rather than dropped.** The
harness's substitution spanned two lines and never matched, so the tree was
unchanged, the build succeeded and `test_ltx2` reported 43/43 SUCCESS — a GREEN
that says nothing about the guarantee. It was caught because the harness prints
`git diff --stat` for every mutation and that line was absent. Re-applied as a
real edit (`1 file changed, 2 insertions(+), 2 deletions(-)`) it is RED. An
unmoved mutation is not evidence of anything.

M3 also shows the doctest counting trap: the case THROWS, so its assertions stop
being counted and the total DROPS 4978 → 4967 while `0 failed` is printed. The
exit code and the `test cases:` line are the authority, not `assertions:`.

### Reachability, checked 2026-08-15 against `AGENTS.md` `## Nothing lands dead`

That rule and [`reachability.md`](../reachability.md) post-date this row
(`8f49ac3be`, [#886](https://github.com/mudler/vllm.cpp/issues/886)), so the
question is answered here rather than assumed. The prompt-side AdaLN is REACHED
from a production entry point on its DEFAULT configuration, and the chain is
`file:line` at this merge commit:

1. `include/vllm.h:962` — `vllm_video_generate`, the shipped C ABI entry point.
2. `src/capi/vllm_c.cpp:1646` — that entry point calls `VideoEngine::Generate` on
   the registry-detected engine.
3. `src/vllm/multimodal/ltx2_video.cpp:1106` — `Ltx2VideoEngine::Generate`.
4. `src/vllm/multimodal/ltx2_video.cpp:1784` and `:1786` — the denoise loop calls
   `Ltx2DitForwardDevice` or `Ltx2DitForward`. Both arms carry the term.
5. `src/vllm/model_executor/models/ltx2_dit.cpp:785` (host) and
   `ltx2_device.cpp:1152` (device) — `prompt_adaln = params.cross_attention_adaln
   && params.use_prompt_adaln_single`, and that flag DEFAULTS TRUE at
   `include/vllm/model_executor/models/ltx2.h:133`. This is the default
   configuration, not an opt-in.
6. `src/vllm/model_executor/models/ltx2_dit.cpp:581` — `PrepareTimestep` runs the
   prompt-side MLP on the stream's own `sigma`.
7. `src/vllm/model_executor/models/ltx2_dit.cpp:205` → `:140` —
   `ModulateContext` adds that row to the static table, on every block of both
   streams.

The loader half is reached the same way: `Ltx2VideoEngine::Load`
(`src/vllm/multimodal/ltx2_video.cpp:575`) calls `Ltx2LoadDitFromSafetensors` at
`:667`, so §3.2's guard runs for every real checkpoint. Both shipped DiTs take
that path with NO opt-in as of #658.

Every anchor above was re-derived at this merge commit and asserted UNIQUE — the
quoted text matches exactly once in its file — against a positive control that
reports `STALE` on a deliberately wrong line. `git grep` alone would not have
answered this; the chain was followed by hand.

**All seven `ltx2_video.cpp` / `test_ltx2_video.cpp` anchors in this section
MOVED under the 2026-08-15 merge of `origin/main`, and the numbers above are the
POST-merge ones.** `0785cfc4d` (#882) added 70 lines to `ltx2_video.cpp` and 306
to `test_ltx2_video.cpp`, ahead of every anchor here: `:1063 → :1106`,
`:1730/:1732 → :1784/:1786`, `:533 → :575`, `:624 → :667`, and in the test
`:1316-1319 → :1620-1625` (widened by one line as well; see `### The gate`) and
`:1302-1303 → :1608-1609`. Each was CORRECT at
`00613767d` and each was WRONG the moment the merge landed, which is the point:
a re-derivation is only true of the tree it ran on, and the merge is part of
landing. They were caught by comparing each span's TEXT against the claim beside
it. A checker that reads the span out of the file and then looks for that span
in the same file is a tautology — it returns unique-and-at-the-cited-line for
every anchor, including the ones now pointing at unrelated code. The expected
text has to come from the CLAIM.

**The second half of the rule is NOT satisfied, and the mutation says so.** The
rule also requires that the smallest failing test ENTER through that entry point.
It does not here. The reachability mutation from `reachability.md` — delete the
production call site, `ltx2_dit.cpp:140` `if (prompt_mod != nullptr) {` becomes
`if (false) {`, so `ModulateContext` ignores the term entirely — was run on a
scratch copy and **BUILT=YES, compile_err=NO**, so this is a test result:

| suite | enters through | result |
|---|---|---|
| `test_ltx2` | `Ltx2DitForward`, by hand | RED — 3 of 43 cases, 6 of 4388 assertions, exit 1 |
| `test_ltx2_video` | `vllm_video_generate` / the ABI | **GREEN** — 37 of 37, 784 of 784, exit 0 (on the previous merge; #882 has since added 3 cases, and the mutation was NOT re-run — the finding is that the suite is green with the term deleted, which more cases cannot undo) |

The entry-point suite drives the path — its fixture sets the flag TRUE
(`tests/vllm/multimodal/ltx2_video_fixture.h:258`) — and asserts no value the
term can move, so it measures that the pipeline runs rather than that this
capability is in it. Filed as
[#900](https://github.com/mudler/vllm.cpp/issues/900) and listed below rather
than repaired here: closing it means designing an ABI-level value oracle
red-first, which is a row with its own spec and not a record repair.

The LOADER half does satisfy both halves. Re-adding
`use_prompt_adaln_single = false` in front of §3.2's guard (M3 below) takes
`test_ltx2_video` to 29 of 37 cases failing, exit 1 — that gate enters through the
production load path and observes the guard.

### Every repo-local citation in this file, re-derived (2026-08-15)

Not only this section's. **33 repo-local citations** were re-derived at the
merge commit that lands: 27 live (25 full-form plus the `:1786`, `:140` and
`:667` bare continuations), 4 SHA-anchored occurrences into `baa92ccf7`
(`ltx2_loader.cpp:988` once, `ltx2.cpp:274-276` three times) and 2 bare
continuations of the SHA-anchored loader claim (`:573`, `:626`). All 33 FRESH.

The rule applied: the expected text is taken from the CLAIM, then required to
occur exactly once in the cited file at the cited revision, beginning on the
cited line. Two consequences worth stating, because each is a finding rather
than a formality:

- `ltx2_loader.cpp:573` and `:626` are the one citation whose text is
  DELIBERATELY not unique — both lines read
  `out.params.use_prompt_adaln_single = false;`, which is exactly what §0 claims
  ("`:573` / `:626` do the same"). Uniqueness is asserted as "exactly these two
  lines", not "exactly one".
- The env-gate anchor was widened from the four gate lines to `:1620-1625` so
  that it resolves at all; see the note in `### The gate`.

The check is armed, not decorative: shifting four anchors by one line
(`ltx2_video.cpp:660`, `:1106`, `ltx2.h:133`, `ltx2.cpp:274-276`) takes it to
6 STALE of 33 — six because the `ltx2.cpp` anchor is cited three times, which is
also why SHA-anchoring it once fixed three sentences.

Upstream citations (`model.py`, `transformer_args.py`, `transformer.py`,
`adaln.py`, `transformer_ltx2.py`, and the rest) are NOT covered by that run.
They are pinned to `fd4ded7f` / `3a2f35d4` and audited separately under
[#794](https://github.com/mudler/vllm.cpp/issues/794).

## Owed

- [#673](https://github.com/mudler/vllm.cpp/issues/673) — this row's
  checkpoint-derived evidence is manual and host-local. `LTX2_CHECKPOINT_ROOT` is
  set by no workflow, so the `test_ltx2_video` shipped-header case skips in CI and
  `scripts/measure-ltx2-prompt-adaln.py` is a manual tool no gate invokes. Filed
  as visible debt; wiring checkpoints into CI is explicitly not in this row.
- [#900](https://github.com/mudler/vllm.cpp/issues/900) — the prompt-AdaLN term is
  REACHED from `vllm_video_generate` (chain above) but no test ENTERS through that
  entry point and observes it: deleting the term's only consumer leaves
  `test_ltx2_video` green at 37 of 37. That is the second half of `AGENTS.md`
  `## Nothing lands dead`, whose rule and guide post-date this row (`8f49ac3be`,
  #886). Closing it needs an ABI-level value oracle designed red-first.
- [#911](https://github.com/mudler/vllm.cpp/issues/911) — this spec is the worked
  example for a class, not an outlier. It shipped EIGHT stale repo-local anchors
  across two repair commits (`7a6165dab`, `00613767d`), every one introduced by
  the row's own commits and moved by its own `020381676` and by `98f8e046d`
  (#658), and then SEVEN more that were correct at `00613767d` and wrong at the
  merge of `origin/main` in this commit. Spec BODIES are checked by nothing:
  `check-agent-record.py`'s `MATRIX_PATHS` (`:521`, `:529-530`) covers the five
  matrices, `feature-matrix.md` and `specs/model-family-inventory.md`, so 4772
  line-carrying citations across 315 `.agents/specs/*.md` are unexamined. The
  rule the issue asks for is one sentence — an anchor into a file the row is
  editing is stale until re-derived at the tree that LANDS, merge included — and
  the two dispositions are already in use here unwritten: `path:NN @ <sha>` for a
  historical claim, claim-sourced uniqueness re-derivation for a live one. Filed
  rather than fixed: a repo-wide sweep and a checker change each need their own
  spec and red-before evidence, and #632's `ENG-RECORD-ANCHOR-RATCHET` is the row
  that should absorb it.

## Now

`DONE` — landed on `row/LTX25-PROMPT-ADALN`.
