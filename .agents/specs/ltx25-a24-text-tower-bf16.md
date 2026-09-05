# LTX-2.5 — the text tower's bfloat16 arm (A24, wave 1 of 8)

Row: `LTX25-A24-TEXT-TOWER-BF16`
Issue: https://github.com/mudler/vllm.cpp/issues/2676
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24)
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `4d10c8acc527c34a6a58a309d52ea5f8fbd1d47b`

## 0. What this row is, in one paragraph

Upstream resolves ONE model dtype for the whole LTX-2.5 pipeline and it is
bfloat16. This tree runs the text tower's aggregation and both caption
projections in f32, and the loader WIDENS a bf16 checkpoint to do it. A24 names
eight components in that shape; this row lands the first of them and leaves the
other seven owed by name. Nothing here is a product decision: upstream's dtype is
the answer, and AGENTS.md forbids asking how a mirrored feature must behave.

The reason this needs its own row rather than a one-line dtype flip is section 4.
Upstream's two normalization variants do NOT compute in the same dtype as each
other on a bf16 input, and neither one computes in the dtype a reader would
guess. That was measured against the pinned oracle, not reasoned about.

## 1. Scope

**IN.** `include/vllm/model_executor/models/ltx2_text_encoder.h`,
`src/vllm/model_executor/models/ltx2_text_encoder.cpp`, the loader entry point
that feeds them (`Ltx2WidenTextProjectionsToF32`, `ltx2_loader.{h,cpp}`), the one
render-path call site that selects the arm (`ltx2_video.cpp`), the golden
generator `scripts/gen-ltx2-text-goldens.py` and the suite
`tests/vllm/models/test_ltx2_text_encoder.cpp`.

**OUT, and owed by name.** The other seven A24 components — the embeddings
connector, the video VAE decoder, the video VAE encoder, the video VAE device
kernels, the tiled-decode buffer, the latent upsampler and the duration head. The
FP8 and NVFP4 arms, which are A22 and are anchored on upstream's quantization
policies rather than on its default dtype. The audio VAE, which §A.7 excludes
deliberately because `ltx2_audio_vae.cpp:7-12` ARGUES its f32 rather than owing
it. The Gemma tower's own hidden-state materialization — see §7.

`.agents/specs/ltx25-completion-scope.md` is operator-owned and this row does not
edit it. §10 records the one anchor drift found in it.

## 2. Upstream anchors, verified in the pinned checkout at this head

| what | anchor | value |
|---|---|---|
| the pipeline's ONE dtype | `ltx-pipelines/.../distilled.py:109` | `self.dtype = torch.bfloat16` |
| handed to `PromptEncoder` | `distilled.py:111-113` | the same object, positionally |
| repeated on the call path | `distilled.py:219` | `dtype = torch.bfloat16` |
| the tower's own default | `gemma/encoders/base_encoder.py:41` | `dtype: torch.dtype = torch.bfloat16` |
| V1 casts BACK before its Linear | `gemma/feature_extractor.py:92,94` | `dtype = encoded.dtype`; `self.aggregate_embed(normed.to(dtype))` |
| V2 casts BACK before its rescale | `gemma/feature_extractor.py:122` | `normed = normed.to(encoded.dtype)` |

Those two `.to(...)` calls are the whole reason §4 exists. A port that assumes
they are no-ops on a bf16 input is wrong about V1 and right about V2 by accident.

## 3. The local side, at `4d10c8acc`

`RequireF32` (`ltx2_text_encoder.cpp:54-62`) refuses every dtype but f32 by name.
It is reached twice:

* `:409`, the first statement of `Ltx2TextFeatureExtractorForward` (`:404`);
* `:546`, the first statement of `Ltx2TextEncoderConditioning` (`:541`), which
  `Ltx2EncodePromptToConditioning` calls at `:1196`, which `ltx2_video.cpp` calls
  from **three** sites. CITED AS A STRING, NOT A LINE NUMBER: this row's own edits
  moved these anchors twice inside one pull request before the citation was
  changed, which is the third recorded drift of the same three numbers.
  `grep -c '= Ltx2EncodePromptToConditioning(' src/vllm/multimodal/ltx2_video.cpp`
  reports exactly 3, and that count is what §1's "reached from a production entry
  point" rests on. (For a reader who wants them: `:2703`, `:3458` and `:6017` at
  this row's landing head — a number that will be wrong again by the next merge.)

That third row is a production entry point on the render path, not a test.

**The checkpoint is ALREADY bf16 and the loader doubles it.**
`Ltx2LoadTextEncoderFromSafetensors` reads both caption projections into
`Ltx2TextProjection::weight_bf16` (`ltx2_loader.h:583-588`).
`Ltx2WidenTextProjectionsToF32` (`ltx2_loader.cpp:1147-1165`) then expands every
element through `Bf16ToF32`, and `ltx2_video.cpp:1789` is what calls it on the
render path. The declaration already prices the widening at
"~4.6 GB at the shipped widths" (`ltx2_loader.h:629-631`) — 3.08 GB for the
[4096, 188160] video projection and 1.54 GB for the [2048, 188160] audio one,
against 1.54 + 0.77 GB of checkpoint bytes.

**The activations pay the same tax twice over.** `Ltx2StackHiddenStates` returns
`[B, T, D, L]` f32; each norm returns `[B, T, D*L]` f32; and V2's `project`
lambda (`ltx2_text_encoder.cpp:461-466`) makes a THIRD full-width copy per
projection for the rescale. At the shipped 3840 x 49 and a 1024-wide padded
prompt that is 770 MB per buffer, and V2 materializes four of them.

## 4. What upstream's bf16 arm actually computes in — MEASURED

Every row below was measured by running the pinned oracle's own functions, not
read off the source. The probe is §9's evidence 1.

### 4.1 V1's normalization returns FLOAT32 on a bf16 input

`_norm_and_concat_padded_batch` builds its mean denominator as
`(sequence_lengths * d) + eps`, and `sequence_lengths` is an int64 tensor. An
int64 tensor plus a Python float promotes to the DEFAULT dtype, so `denom` is
float32; `bf16_sum / f32_denom` promotes the mean to float32; and every
subsequent term inherits it. Measured: `V1 out dtype: torch.float32` for a bf16
input.

That is why `:94` writes `self.aggregate_embed(normed.to(dtype))`. The cast is
not decoration, it is the ONLY place V1 narrows, and a port that runs V1's norm
in bf16 arithmetic diverges from upstream everywhere.

The narrowing that does happen inside V1 on a bf16 input, and that a naive f32
implementation would miss:

* `masked.sum(dim=(1,2))` accumulates in f32 and ROUNDS TO BF16 before the
  division (measured `masked.sum dtype: torch.bfloat16`);
* `range_ = x_max - x_min` is a bf16 subtraction and rounds;
* `range_ + eps` is bf16 (see §4.3).

### 4.2 V2's normalization is bf16 throughout, and squares BEFORE it accumulates

Measured `V2 out dtype: torch.bfloat16`, and the discriminating fact:

```
var bitmatch f32acc-of-bf16-squares:               True
var bitmatch f32-squares (no bf16 round of square): False
```

`encoded_text**2` materializes a bf16 tensor, so each square is rounded to 8
mantissa bits BEFORE the mean accumulates it. An implementation that squares in
f32 and accumulates in f32 is bit-wrong, and this is the bit that says so.

`torch.mean` over a bf16 tensor accumulates in f32 (`opmath_type`), divides in
f32, and rounds the result to bf16.

### 4.3 The epsilon that reaches the bf16 arm is `bf16(1e-6)`, not `1e-6`

`variance + 1e-6` pairs a bf16 tensor with a Python float. Measured over all
32639 finite non-negative bf16 values:

```
add-scalar == bf16(f32 add of 1e-6):        False
add-scalar == bf16(f32 add of bf16(1e-6)):  True
```

`bf16(1e-6) = 9.98377799987793e-07` (`0x3586`). This is the same class of trap the
file header already names for the f32 arm ("the epsilons, which are a class and
not one instance"), one width down. The V1 range epsilon takes the same route,
because `range_` is bf16.

### 4.4 Every bf16 elementwise op is "compute in f32, round to nearest even"

Measured exhaustively over the bf16 domain: `mul`, `sub`, `pow(2)` and `rsqrt`
all equal `bf16(f32 op)` on every input tried, with zero mismatches, and the
narrowing is round-to-nearest-EVEN (`1.0 + 2**-9 -> 0x3f80`,
`1.0 + 3*2**-9 -> 0x3f81`). `vt::F32ToBF16` (`src/vt/dtype.cpp:306-313`) is
already RNE, so the tree's own converter is the right one.

### 4.5 `nn.Linear` accumulates in f32 and rounds ONCE, bias included

Measured bit-exact: `linear(x) == (x.f32 @ W.f32.T + b.f32).to(bf16)`, maxdiff
0.0. The bias is added BEFORE the narrowing. A port that rounds the GEMM output
to bf16 and then adds a bias rounds twice and is wrong.

`vt::MatmulBT`'s CPU kernel already has exactly this contract: `acc` is f32,
`LoadF32` widens a bf16 operand exactly, and `StoreF32` rounds on the way out
(`src/vt/cpu/cpu_ops.cpp:151-180`). So the bf16 Linear is bf16 operands into an
**f32** output tensor, an f32 bias add, and one `F32ToBF16` — no new kernel.

### 4.6 The rescale factor is the bf16-rounded one

`_rescale_norm` multiplies by a Python float, so §4.3's rule applies: the factor
that multiplies a bf16 activation is `bf16(f32(sqrt(target/source)))`. Measured
1.15625 for the fixture's 8/6. `Ltx2RescaleNorm` already returns the f32-rounded
double for the same reason one width up, so the bf16 arm narrows its return value
and nothing else changes.

## 5. Design

### 5.1 The weights carry their own dtype, as upstream's parameters do

`Ltx2TextAggregateEmbed` gains `weight_bf16` / `bias_bf16` (`uint16_t`) and a
`dtype` field. Exactly one of the two storages is populated. This is the memory
format the row exists to change, so it is a storage change and not a flag.

`Ltx2TextEncoderWeights::ComputeDtype()` reports the arm the projections carry and
refuses a video/audio disagreement by name.

### 5.2 The refusal narrows instead of disappearing

`RequireF32` becomes `RequireTextComputeDtype(compute_dtype, weights)`. It

* accepts `kF32` and `kBF16`;
* refuses every other dtype BY NAME, and the message points FP8 / NVFP4 at A22
  rather than at the retired phase-L6 note;
* refuses a `compute_dtype` that disagrees with what `weights` carries.

The third clause matters more than the first two. It is the same
declared-versus-actual discipline `RequireDeclaredProjection` already applies to
`aggregate_bias` and `out_features`, and it is what stops a caller from selecting
an arm the weights cannot serve.

### 5.3 The bf16 compute path

`Ltx2StackHiddenStatesBf16`, `Ltx2NormAndConcatPaddedBatchBf16` and
`Ltx2NormAndConcatPerTokenRmsBf16` return `std::vector<uint16_t>`, halving every
full-width buffer in §3. Their arithmetic mirrors §4 exactly:

* V1 computes its mean, range and normalization in **f32** (§4.1), rounding to
  bf16 only at `masked.sum`, at `range_`, at `range_ + eps` and once at the end;
* V2 computes in bf16 throughout, squaring to bf16 before the f32-accumulated
  mean (§4.2);
* both add `bf16(1e-6)` (§4.3);
* the projection is `vt::MatmulBT` with bf16 operands and an f32 output, an f32
  bias add, and one narrowing (§4.5).

The f32 arm is untouched, byte for byte. It stays the declared PARITY arm that
the existing 27 cases gate, and the two arms share no arithmetic, so neither can
silently become the other.

### 5.4 The production path selects bf16 because the checkpoint is bf16

`Ltx2TextProjectionsAsBf16(checkpoint)` moves the checkpoint's own bytes into
`Ltx2TextEncoderWeights` without widening. `ltx2_video.cpp:1789` calls it instead
of `Ltx2WidenTextProjectionsToF32`, and `Ltx2EncodePromptToConditioning` takes its
compute dtype from `weights.ComputeDtype()`. No render call site grows a
parameter, and the arm follows the weights the way upstream's does.

`Ltx2WidenTextProjectionsToF32` stays. It is what the f32 parity arm and the
`probe_ltx2_text_encoder_load` probe use, and deleting it would delete the
reference arm the goldens are measured against.

### 5.5 The boundary that stays f32, and why it is owed rather than hidden

`Ltx2TextConditioning` keeps `std::vector<float>` fields. Its only consumer is
`Ltx2ConnectorForward`, which is A24's SECOND wave and is f32 throughout; making
the struct bf16 here would change a component this row excludes. So the tower
computes in bf16 and materializes bf16 VALUES in an f32 container at the seam —
which §6.4's assertion turns into the row's strongest gate rather than a soft
spot — and the container is listed under `## Owed` against the connector wave.

## 6. Tests

### 6.1 RED FIRST, through a production entry point

`Ltx2EncodePromptToConditioning` is the tower entry point the render calls, and
the existing suite already reaches it with a fixture tower. The new cases enter
there, with bf16 projections. Before the implementation they red on
`RequireF32`'s own message.

No case constructs the extractor by hand to prove the class works.

### 6.2 The goldens come from upstream RUN IN BF16

`scripts/gen-ltx2-text-goldens.py` already imports the oracle by path, asserts its
identity, and executes it on CPU at reduced dimensions. It gains a bf16 section
that runs the SAME `FeatureExtractorV1` / `FeatureExtractorV2` modules, cast with
`.to(torch.bfloat16)`, on bf16 inputs — every parameter, tolerance and failure
case of the f32 section preserved, one dtype changed. No GPU, no checkpoint.

The section also emits the four §4 discriminators as goldens in their own right:
the V1 norm's output dtype, the bf16-rounded epsilon, the bf16 rescale factor,
and the bf16-squares variance.

### 6.3 The value gate

Bit-exactness is attempted first and used where it holds, because it is free and
strictly stronger. Where the GEMM's K-reduction order prevents it, the bound is
bf16 unit roundoff (`2^-8` relative) against the golden's own magnitude —
derived from the format, not fitted to the result. Any case that needs the looser
bound records its measured margin beside it.

### 6.3b The epsilon's WIDTH, which the value goldens cannot see

`kLtx2TextNormEpsBf16` is a claimed guarantee, and a golden built on an
ordinary or a degenerate variance cannot hold it: at bf16 the epsilon is eight
orders below the format's resolution, so on almost every input `var + bf16(eps)`
and `var + f32(eps)` narrow to the same bits. The generator therefore sweeps the
whole bf16 domain for the values at which they do NOT — 140 of 32640 at the pin,
all inside `[0x384a, 0x3b25]` — lays one per `(b, t, layer)` slice, and emits
upstream's answer beside the REJECTED f32-scalar one. It refuses to emit the
probe at all if the two ever stop parting. The suite holds the port bit-exact to
the first and asserts the second differs, in the same shape the rescale probe
already uses.

### 6.4 The DTYPE gate, which is the one that cannot be faked

A token gate cannot see a dtype that is too wide, so the row is gated on the
memory format directly:

1. **The weights.** On one synthetic checkpoint, `Ltx2TextProjectionsAsBf16` and
   `Ltx2WidenTextProjectionsToF32` are both materialized and their byte counts
   compared: the bf16 arm must be EXACTLY half, and its f32 storage must be
   empty. Measured on the same input, so no number is quoted.
2. **The intermediates.** `sizeof(value_type) == 2` and the byte count of each
   full-width buffer, against the f32 arm's on the same shapes.
3. **The output, and this is the assertion an f32 path cannot pass.** Every float
   `Ltx2EncodePromptToConditioning` returns under bf16 weights must satisfy
   `BF16ToF32(F32ToBF16(v)) == v` — it must survive a bf16 round trip unchanged.
   An f32 path fails this on essentially every value, which the same case asserts
   in the other direction so the gate is proven to discriminate rather than
   assumed to.

(3) is what makes the goldens meaningful. A golden alone passes in both arms.

### 6.5 The mutation a fresh reviewer applies

* Delete `ltx2_video.cpp:1789`'s `Ltx2TextProjectionsAsBf16` call and restore the
  widening call. The loader case must red.
* Change the bf16 epsilon from `bf16(1e-6)` to `1e-6`. The V2 case "an input
  on which the epsilon's own WIDTH is visible" must red. Its probe is the one
  §4.3's value goldens are NOT: the tiny-variance and zero-variance inputs
  discriminate against a dropped epsilon and a 100x one and read identically
  under the two widths, and the transcription check on `kLtx2TextNormV2Eps`
  compares one constant against another without reaching the arithmetic. Only
  the 140 swept values on which the two narrowings part can red this.
* Square in f32 instead of rounding to bf16 first. §4.2's golden must red.
* Add the Linear's bias after the narrowing instead of before. §4.5 must red.
* Delete the round-trip assertion in §6.4(3) and run the bf16 cases against the
  f32 arm. The value goldens alone must NOT be enough to red it — that is the
  measurement that proves (3) is load-bearing.

## 7. Risks

* **The Gemma tower still materializes f32 hidden states.**
  `Gemma4Model::ForwardHiddenStates` returns `std::vector<std::vector<float>>`,
  and `Ltx2TextHiddenStates` holds `const float*`. This row narrows them at the
  stack, which is where upstream's own tensor is already bf16, so the arithmetic
  mirrors. What it does NOT do is stop the tower from holding 49 f32 states. That
  is `gemma4.*`, not `ltx2_text_encoder.*`, and it is owed.
* **bf16 is lossy and this row makes the render less precise than it was.** That
  is the point: upstream's answer is the bf16 one, and an f32 answer that is
  "better" is a divergence. The f32 arm remains reachable and gated.
* **A K = 188160 reduction in f32 accumulation.** Unchanged by this row — the
  accumulator width is the same as the f32 arm's, only the operands narrow — but
  it is the reason the bf16 GEMM keeps an f32 output tensor rather than
  accumulating in bf16.

## 8. Gates

```sh
# G1 — the refusal count, before and after. It NARROWS; it must not vanish.
grep -n "compute_dtype must be" src/vllm/model_executor/models/ltx2_text_encoder.cpp

# G2 — the render path selects the bf16 arm, and there is exactly one selector.
grep -n "Ltx2TextProjectionsAsBf16\|Ltx2WidenTextProjectionsToF32" \
  src/vllm/multimodal/ltx2_video.cpp

# G3 — the goldens are upstream's, regenerated at the pin.
python3 scripts/gen-ltx2-text-goldens.py --ltx2 ~/_git/LTX-2 \
  --out tests/vllm/models/ltx2_text_goldens.inc && git diff --stat

# G4 — the focused suite, both arms.
ctest --test-dir build -R ltx2_text --output-on-failure

# G5 — the full gate.
scripts/agent-preflight.sh --staged
python3 scripts/check-pr-size.py --base origin/main --head HEAD
```

## 9. Evidence

1. The dtype probes of §4, run against `fd4ded7f` with torch 2.11.0+cu130 on
   CPU. Recorded in `## Outcome` with their literal output.
2. The regenerated `ltx2_text_goldens.inc`, whose header carries the upstream
   revision the generator read from git.
3. The measured byte counts of §6.4(1), printed by the case itself.

## 10. A contradiction found in the parent scope document

`.agents/specs/ltx25-completion-scope.md` §A.7 gives the render-path call sites
of `Ltx2EncodePromptToConditioning` as `ltx2_video.cpp:2501`, `:3241` and `:5798`.
The COUNT and the file are right and the row's conclusion is unaffected; the line
numbers drifted after the scope document was written. That document is
operator-owned, so this row records the drift here rather than editing it.

And then this row drifted its own replacement, twice: it first wrote `:2683`,
`:3424` and `:5983` — the SECOND line of each call, read at base `4d10c8acc` where
the calls begin at `:2682`, `:3423` and `:5982` — and its own later commits moved
them again, to `:2703`, `:3458` and `:6017`. Three drifts of the same three
numbers inside one row is the argument, so §3 and the suite now cite the call
STRING and its count instead, which no edit above them can move. This is the
"recorded line anchors go stale" failure, and it is why §2 and §3 above were
re-read at this head.

## Outcome

Everything below was measured on this branch, not predicted by §4.

### What §4 got right, and the one thing it got wrong

Facts 1, 2, 4 and 5 held exactly as written. Fact 3 — "a Python float paired with
a bf16 tensor is narrowed first" — is **true of `add` and false of `mul`**, and
the first implementation of this row got the rescale wrong because of it.

Measured exhaustively over the bf16 domain at the pin:

| op | bf16-narrowed scalar | f32 scalar |
|---|---|---|
| `t + 1e-6` | matches at **all 32640** | matches at all but 387 |
| `t * sqrt(8/6)` | matches at all but 7881 | matches at **all 32639** |

So the epsilon really is `bf16(1e-6)` and the rescale factor really is not
narrowed. Two scalar ops one line of upstream apart, two answers, and no single
rule covers both.

**The probe that let the defect through multiplied by ONE.** `_rescale_norm(ones,
...)` narrows to `bf16(f)` under both hypotheses, so its golden agreed with a
port that was wrong on nearly a quarter of every other value. The probe is now a
vector chosen to separate them — two values found by sweeping the exponent range
against each factor — and the generator REFUSES to emit it if it ever stops
separating them. Fixing this cut the V2 projection's divergence from 36/80 values
to 10/80.

### A fifth fact, which changed the gate's shape

**`torch.rsqrt` on bf16 is not a function of its input**, and not marginally so.
The same value gives `0x4065` in a length-1 tensor and `0x4066` in a length-1000
one, because the vectorized body and the scalar tail round differently — and that
is not one unlucky value: swept at the pin, **9027 of the 32639 finite positive
bf16 values** read differently in a length-1 tensor than in the vectorized one.
`0x4066` is the correctly rounded answer and is what this port computes.

No implementation can be bit-equal to a kernel that is not bit-equal to itself,
and chasing one would fit this machine's SIMD width rather than upstream's
arithmetic. So each bf16 golden is emitted TWICE: once from the module, and once
from the module with `torch.rsqrt` replaced by the CORRECTLY ROUNDED value,
`bf16(1.0 / sqrt(f32(x)))`. That is not the same statement as "the value its own
vectorized path computes", which an earlier draft of this section made: the two
agree on all but 6 of the 32639 values, and the six are exactly where the
phrasing would have been wrong. The port is held **bit-exact** to the correctly
rounded oracle — no tolerance sits under this port's arithmetic — and its
distance to the unpatched module is reported.

§6.3 planned "bit-exact where it holds, bf16 unit roundoff otherwise". What
landed is better: bit-exact everywhere against a stable oracle, plus a reported
distance to the unstable one.

### The two bounds that are not bit-exact, and where each number comes from

* **V2 norm vs the unpatched module: 2 ulp on at most `self-disagree x D`
  values.** The norm ends in `x * inv`, and `|x*inv1 - x*inv2| = |x| * ulp(inv)`
  is 2^-8 relative to the product — which is TWO ulp of a product at the top of
  its binade. The count budget is the generator's own measurement of how many
  variance entries torch disagrees with itself on (2 of 40), times the D values
  one rsqrt feeds. Measured: 11 and 5 of 240, worst 2 ulp.
* **Projection outputs vs the unpatched module: absolute, `2 x 2^-8 x
  max|golden|`.** Not per-element relative, and the reason is arithmetic: each
  output is a dot of 24 terms, so one that lands near zero through cancellation
  moves arbitrarily far in relative terms on a last-bit input change. Measured
  worst 0.00195 against a 0.00327 bound.
* **The two arms against each other at the tower level: `sqrt(flat) x 2^-9 x
  max|f32|`,** the random-walk accumulation of one bf16 unit roundoff over a
  `flat`-term reduction. Measured 0.00844 against 0.0615 — a 7.3x margin, so the
  bound is not doing the work the assertion claims for it.

### The dtype gate, which is the row

`ltx2 text bf16: the production entry point computes in bf16` reports:

```
bf16 arm: 0 of 480 video values are wider than bf16; f32 arm: 480
caption projections: bf16 576 B, f32 1152 B
```

Perfect discrimination, both directions, in the same case.

At the ENGINE level, `Ltx2ConditioningTrace` gained `tower_video_not_bf16` /
`tower_audio_not_bf16`, sampled after the tower and BEFORE the connector, and
`test_ltx2_video`'s prompted-render case asserts they are zero.

**The mutation §6.5 names was run.** Reverting the engine's
`Ltx2TextProjectionsAsBf16` call to `Ltx2WidenTextProjectionsToF32`:

```
ERROR: CHECK( fox.trace.tower_video_not_bf16 == 0 ) is NOT correct!
  logged: tower output wider than bf16: video 64 of 16384, audio 32 of 8192
[doctest] assertions: 63 | 59 passed | 4 failed |
```

Exactly the four new assertions red. The other 59 — the digests, the absmax, the
prompt dependence, the frame bytes and the determinism check — stay green to the
last one. That is simultaneously the reachability proof and the demonstration of
why A24 was invisible: every gate that existed on this path passes under the
defect. The tree was restored byte-for-byte and re-verified green afterwards.

### Why the counters are sampled before the connector

Measuring after it reports **16384 of 16384** wider than bf16 even on a bf16
tower, because `Ltx2ConnectorForward` is A24 wave 2 and still computes in f32.
That was measured, not assumed, and it is the sharpest available statement of
what this row does and does not deliver.

## Owed

* **The other seven A24 components**, none of which this row touches: the
  embeddings connector (`ltx2_connector.h:47-51`), the video VAE decoder
  (`ltx2_video_vae.h:47-54`), the video VAE encoder
  (`ltx2_video_vae_encoder.h:52-57`), the video VAE device kernels
  (`ltx2_video_vae_kernels.h:44-51`), the tiled-decode buffer
  (`ltx2_tiling.h:88-94`), the latent upsampler (`ltx2_upsampler.h:66-70`) and
  the duration head (`ltx2_duration_head.h:55-58`).
* **`Ltx2TextConditioning`'s f32 container** (§5.5). It narrows when the
  connector wave lands, because both sides have to move together.
* **The Gemma tower's f32 hidden states** (§7). `Gemma4Model::ForwardHiddenStates`
  returns 49 f32 buffers where upstream holds bf16. Outside this row's files.
* **The FP8 and NVFP4 arms**, which are A22 and are not a dtype default.
* **A real-weights render in bf16 against upstream**, which needs a GPU lease and
  a checkpoint. This row gates the arithmetic against upstream executed on CPU at
  reduced dimensions; it does not claim a full-render token gate.

## Now

`DONE`. The spec commit precedes the implementation commit, which is the commit
order that proves it came first.
