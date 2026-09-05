# LTX-2.5 — the embeddings connector's bfloat16 arm (A24, wave 2 of 8)

Row: `LTX25-A24-CONNECTOR-BF16`
Issue: https://github.com/mudler/vllm.cpp/issues/2720
Wave 1: `.agents/specs/ltx25-a24-text-tower-bf16.md` (#2676, PR #2681, `8e582a5f9`)
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24)
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `8e582a5f9`

## 0. What this row is, in one paragraph

Upstream constructs `Embeddings1DConnector` inside `PromptEncoder`, which is
handed the pipeline's ONE dtype, `torch.bfloat16` (`distilled.py:109`, `:113`).
This tree widens the connector's bf16 checkpoint tensors to f32 at load and
computes every activation in f32. Wave 1 landed the text tower; this row lands
the connector, which is the second of A24's eight components and the second of
the two host leaves that are 54.03% of the render wall. The other six stay owed
by name. Nothing here is a product decision.

The reason it needs a row rather than a dtype flip is §4. Upstream's bf16
connector does not round where a reader would guess, and two of its own rope arms
— one line apart in the same file — round DIFFERENTLY from each other. Every fact
in §4 was measured by executing the pinned module, not read off it.

## 1. Scope

**IN.** `include/vllm/model_executor/models/ltx2_connector.h`,
`src/vllm/model_executor/models/ltx2_connector.cpp`; the loader entry point that
feeds them (`Ltx2LoadConnectorWeights`, `ltx2_loader.{h,cpp}`) and the bf16
storage `Ltx2VaeWeights` needs to hold the arm; the two shared seams the
connector routes through (`Ltx2Attention`, `Ltx2FeedForward`, `Linear`,
`Ltx2ApplyRotaryEmb` in `ltx2.{h,cpp}`), extended rather than duplicated; the
render-path call sites in `src/vllm/multimodal/ltx2_video.cpp` that select the
arm; the golden generator `scripts/gen-ltx2-pipeline-goldens.py` section 10 and
the suite `tests/vllm/models/test_ltx2_pipeline.cpp`; and the engine-level dtype
counters wave 1 installed in `tests/vllm/multimodal/test_ltx2_video.cpp`.

**OUT, and owed by name.** The other six A24 components — the video VAE decoder,
the video VAE encoder, the video VAE device kernels, the tiled-decode buffer, the
latent upsampler and the duration head. The FP8 and NVFP4 arms, which are A22.
The audio VAE, whose f32 `ltx2_audio_vae.cpp:7-12` ARGUES rather than owes. The
Gemma tower's 49 f32 hidden-state buffers, which wave 1 established are a memory
debt only.

`.agents/specs/ltx25-completion-scope.md` is operator-owned and this row does not
edit it.

## 2. Upstream anchors, verified in the pinned checkout at this head

| what | anchor | value |
|---|---|---|
| the pipeline's ONE dtype | `ltx-pipelines/.../distilled.py:109` | `self.dtype = torch.bfloat16` |
| handed to `PromptEncoder` | `distilled.py:111-113` | the same object, positionally |
| the connector is built INSIDE it | `text_encoders/gemma/prompt_encoder.py` | `Embeddings1DConnector` is a `PromptEncoder` submodule, so it inherits that dtype |
| the register table is bf16 EVEN AT f32 | `embeddings_connector.py:135-137` | `torch.rand(..., dtype=torch.bfloat16)` |
| the RoPE tables take the ACTIVATION dtype | `embeddings_connector.py:176` + `rope.py:224` | `out_dtype=hidden_states.dtype`; `cos_freq.to(out_dtype)` |
| split rope's second term is FUSED | `rope.py:75-76` | `addcmul_(-sin, second_half_input)` |
| interleaved rope's is NOT | `rope.py:38` | `input * cos + input_rot * sin` |
| the weightless final norm | `embeddings_connector.py:189`, `utils.py:7-12` | `rms_norm(x)`, `eps=1e-6` |

The last two rows are §4.3 and §4.4, and they are why this row exists.

## 3. The local side, at `8e582a5f9`

`Ltx2LoadConnectorWeights` (`ltx2_loader.cpp:1534`) materializes every enumerated
tensor and, when the checkpoint holds BF16, expands each element through
`Bf16ToF32` into a `std::vector<float>`. Its own declaration prices the family at
"about 8 GB of f32 together" (`ltx2_loader.h:757-766`) for ~2.016 B parameters,
against ~4 GB of checkpoint bytes.

`Ltx2ConnectorForward` computes f32 throughout and says so
(`ltx2_connector.h:47-51`, the `DTYPE` block): "f32 for the activations, which is
the parity dtype of this gate ... The production bf16 arm is phase L6." That is a
HEADER COMMENT and not a refusal, which is exactly why §A.7's refusal sweep could
not see it.

**Reached from a production entry point.** CITED AS A STRING, NOT A LINE NUMBER,
because wave 1 recorded three drifts of the same three numbers in
`ltx2_video.cpp` inside one pull request:

```sh
grep -c 'RunConnector(' src/vllm/multimodal/ltx2_video.cpp         # 6
grep -c 'RunConnectorFromFile(' src/vllm/multimodal/ltx2_video.cpp  # 4
```

**Those two numbers were WRITTEN before they were run, and the first one was
wrong.** This section first said "5: 2 definitions + 3 calls" for `RunConnector`
on no measurement at all, which is the same error as citing a line number nobody
re-read. Run at this head, the decomposition is: `RunConnector` has **two
overloads**, one taking materialized weights and one taking a `ConnectorWeightSet`;
the second forwards to the first, and `RunConnectorFromFile` forwards to it too.
So of the six, two are definitions, two are those internal forwards, and **two are
render-path calls**. `RunConnectorFromFile` is 1 definition and 3 calls, which is
what it said.

Both are `Ltx2VideoEngine` functions on the render path: the prompted render
reaches `RunConnector` through the `ConnectorWeightSet` overload, and the
prompt-embeds render reaches `RunConnectorFromFile`. Every one of the four
materialization sites now asks for `kBF16`, which
`grep -c 'Ltx2LoadConnectorWeights(.*kBF16'` reports as 4.

**Wave 1 already built this row's instrument and measured its red.**
`Ltx2ConditioningTrace` (`multimodal/ltx2_video.h`) carries
`connector_video_not_bf16` / `connector_audio_not_bf16`, sampled on the connector
OUTPUT in the same render as the tower counters, and
`test_ltx2_video.cpp`'s prompted-render case asserts they are MOST of the stream
— the f32 population that proves the tower's own `== 0` is not vacuous. Wave 1's
`## Outcome` records the value: 16384 of 16384 and 8192 of 8192.

## 4. What upstream's bf16 connector actually computes in — MEASURED

Every row below was produced by running the pinned modules on CPU with torch
2.11.0+cu130, not by reading them. §9 evidence 1 carries the probes.

### 4.1 `rms_norm` on bf16 is computed ENTIRELY in f32 and rounded ONCE

`torch.nn.functional.rms_norm(bf16)` returns bf16 and is bit-equal to

```
bf16( x.float() * rsqrt( mean(x.float() * x.float()) + 1e-6_f32 ) )
```

over 1 973 760 values: **0 mismatches**. The three neighbouring hypotheses are
distinguishable and all wrong — `x/sqrt(...)` differs on 1, an f64 sum differs on
1, and the bf16-narrowed epsilon differs on 4.

This is the OPPOSITE of wave 1's V2 tower norm, which squares to bf16 BEFORE it
accumulates. Two norms in the same pipeline, two rules. It is also STABLE under
shape, which wave 1's bf16 `rsqrt` was not: isolating each of 512 rows and
re-normalizing it alone reproduces the batched answer on **512 of 512**.

`torch.nn.RMSNorm` WITH a weight (the q/k norms) is bit-equal to the same
expression times the f32-widened weight, 0 of 491 520.

> **RETRACTED, 2026-09-03.** This paragraph originally said the alternative —
> round the normalized value to bf16 and THEN multiply by the gain — was "NOT
> separable from it on that sweep (0 mismatches either way)", and that reasoning
> is what justified not gating the alternative. Executed, it separates cleanly.
> Measured on this branch with the suite's own 1-D `.weight` fixture at width 24:
> upstream against the fused form **0**, against round-then-multiply **12 647 of
> 49 152** at ordinary magnitude and **3 166 of 12 288** at the probe's 2^-13
> rows. The sweep that reported 0 either way did not measure what this sentence
> claimed. What actually holds is the first sentence alone: upstream is the FUSED
> f32 form, which is `F.rms_norm`'s own structure, and the round-then-multiply
> alternative is a distinguishable and wrong one. It is now emitted as a second
> rejected hypothesis beside upstream's answer in the weighted epsilon probe
> (`kLtx2ConnBf16EpsWeightedRoundThenMul`), so the suite asserts the difference
> rather than recording that it could not see one.

### 4.2 The epsilon is the f32 `1e-6`, and it only shows on SMALL rows

Wave 1's tower epsilon is `bf16(1e-6)` because it is added to a bf16 tensor.
Here it is added inside an f32 accumulator, so it is `1e-6` exactly. Measured
against `bf16(1e-6) = 9.98377799987793e-07`:

| row scale | values where the two epsilons part | f32-eps mismatches | bf16-eps mismatches |
|---|---|---|---|
| 2^-6 | 315 / 160000 | **0** | 315 |
| 2^-4 | 26 / 160000 | **0** | 26 |
| 2^-2 | 3 / 160000 | **0** | 3 |
| 2^-1 and above | 0 / 160000 | 0 | 0 |

**The probe therefore has to be built to separate.** On rows of ordinary
magnitude the two epsilons are indistinguishable, so a golden taken on the
existing fixture gates nothing about the epsilon — the wave-1 failure, one width
down. The generator lays SMALL-MAGNITUDE rows for this probe, emits upstream's
answer beside the REJECTED bf16-scalar one, and refuses to emit at all if the two
stop parting.

### 4.3 Split RoPE FUSES its second term; interleaved RoPE does not

Split (`rope.py:69-76`) multiplies both halves by `cos`, materializing a bf16
tensor, and then applies `addcmul_`, which fuses the second multiply into the add
with ONE rounding:

```
out_first  = bf16( bf16(t1*cos) + (-sin * t2) )
out_second = bf16( bf16(t2*cos) + ( sin * t1) )
```

Measured **0 of 384** against upstream. The two natural alternatives are wrong
and the probe separates them: "all f32, one round" differs on 92 of 384, "every
op rounded separately" on 101.

Interleaved (`rope.py:38`) is a plain `a*cos + b*sin` and rounds EVERY op:

```
out = bf16( bf16(t*cos) + bf16(rot*sin) )
```

Measured **0 of 384**; "all f32, one round" differs on **141 of 384**.

So the two rope arms round differently, they are six lines apart, and no single
rule covers both. This is wave 1's add/mul asymmetry in a new place.

### 4.4 The RoPE TABLES themselves narrow to bf16

`precompute_freqs_cis(out_dtype=hidden_states.dtype)` ends in
`cos_freq.to(out_dtype)` (`rope.py:224`), so on a bf16 activation the cos/sin
tables are bf16 — and measured bit-equal to `bf16(the f32 tables)`, so the port
computes the tables as it does today and rounds them. The `double_precision_rope`
arm computes its grid in float64 and lands in the same `.to(out_dtype)`.

### 4.5 `nn.Linear` accumulates in f32 and rounds ONCE, bias included

Bit-exact: `linear(x) == bf16(x.f32 @ W.f32.T + b.f32)`. Same contract wave 1
measured, and `vt::MatmulBT`'s CPU kernel already has it —
`MatmulOneChunkRef` keeps `acc` in f32, `LoadF32` widens a bf16 operand exactly,
and `StoreF32` writes the output tensor's own dtype
(`src/vt/cpu/cpu_ops.cpp:151-180`). So a bf16 Linear is bf16 operands, an f32
output tensor, an f32 bias add, and one `F32ToBF16`. No new kernel.

### 4.6 SDPA at bf16 is NOT a function of its input, and MATH is the oracle

`AttentionFunction.AUTOMATIC` resolves to
`SDPA[FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH>CUDNN_ATTENTION>OVERRIDEABLE]` and
on this CPU the FLASH kernel serves the call. **No formula reproduces it.** Eight
hypotheses over {round scores, round probabilities, scale before/after} were
measured against it and the best still differs on 128 of 384 bf16 words; a
fully-materializing bf16 chain differs on 195.

`SDPBackend.MATH` IS well-defined: it is bit-equal to
`bf16(f32-accumulated attention with no intermediate rounding)` — **0 of 384**
against both an f32 and an f64 reference, so the accumulation is exact enough
that the two agree.

This is wave 1's `torch.rsqrt` finding at the kernel level, and it takes wave 1's
remedy: each bf16 golden is emitted TWICE, once from the module as constructed
(FLASH) and once with both attention callables pinned to `SDPA_MATH`. The port is
held to the MATH oracle and its distance to the unpatched module is reported.

**The f32 arm already lives with this**, which is why it is a tolerance question
and not a defect: at f32 the same two backends differ by 4.77e-7 across the whole
connector, and the existing five arms pass at `kRoundOff`. At bf16 they differ by
0.015625 against a max|golden| of 2.765625 — two bf16 ulps at that binade, i.e.
the format's own resolution rather than an error.

### 4.7 The register substitution is a SELECT at bf16, and the table stops moving

`registers.to(hidden_states.dtype)` is bf16 -> bf16, a no-op; `binary_mask` is
exactly 0 or 1, so `mask*h + (1-mask)*r` is exact at any width. Measured: the
substitution returns bf16 and the zeroed mask stays **f32** — the mask is
`torch.zeros_like(additive_attention_mask)` and the caller's mask is f32, so the
mask does NOT narrow with the activations. A port that narrowed it would be wrong
about a tensor that never sees the model dtype.

The f32 arm's `RoundToBf16` on the table becomes an identity at bf16, because the
stored value already is the bf16 one. That is the same number by two routes, and
it is the one place the two arms provably agree.

### 4.8 GELU-tanh and the residual adds

`GELUApprox` is `F.gelu(self.proj(x), approximate="tanh")`, and at bf16 it is
bit-equal to the f32 tanh formula applied to the bf16 Linear output and rounded
once (0 of 6144). Every bf16 add is `bf16(f32 add)`.

## 5. Design

### 5.1 The weights carry their own dtype

`Ltx2VaeWeights` gains `bf16` storage (`std::map<std::string, std::vector<uint16_t>>`)
beside its `tensors`, and a `dtype`. Exactly one map is populated. This is the
memory format the row exists to change, so it is a storage change and not a flag.
`Ltx2LoadConnectorWeights` gains a `vt::DType` parameter: at `kBF16` it keeps the
checkpoint's own 16-bit words instead of expanding them, halving the family from
~8 GB to ~4 GB. A checkpoint that stores F32 under a bf16 request is narrowed
once at load, which is what upstream's `.to(dtype)` does to a module built from an
f32 state dict.

### 5.2 The shared seams are EXTENDED, never duplicated

`ltx2_connector.h` already forbids a second attention here — "a second attention
implementation here would be a parallel path that could drift from the one the
DiT is gated on" — and AGENTS.md says the same. So:

* `Linear` (`ltx2.cpp:31`) accepts a `kBF16` weight tensor and keeps refusing
  every other dtype by name. `vt::MatmulBT` already serves the mixed case (§4.5).
* `Ltx2AttentionArgs` gains `vt::DType compute_dtype = kF32`, and
  `Ltx2FeedForward` gains the same parameter with the same default. At `kBF16`
  each function narrows exactly the intermediates upstream materializes and
  nothing else. At `kF32` — every existing caller — not one float moves.
* `Ltx2ApplyRotaryEmb` gains the dtype too, because §4.3's two arms round
  differently and the rounding cannot be applied from outside the function.

The default keeps the DiT's reference arm byte-identical, and the two arms share
no rounding, so neither can silently become the other.

### 5.3 What the connector's bf16 arm computes

`Ltx2ConnectorForward` and `Ltx2ConnectorReplaceRegisters` gain a
`compute_dtype`, resolved from the weights. At `kBF16`:

* the register table is used as stored, and the output of the substitution is
  bf16-valued (§4.7);
* the RoPE tables are computed as today and narrowed (§4.4);
* every `rms_norm` widens to f32, adds the f32 `1e-6` — not `bf16(1e-6)` — and
  rounds the result once (§4.1, §4.2). **The mean-square accumulator STAYS f64 on
  both arms**, which is the opposite of what this line predicted before the work
  ran: the branch measured `torch.mean` to be a BLOCKED f32 reduction that an
  exact f64 sum approximates far better than a same-width sequential one (11 000
  mismatches against 24 at the shipped width 3840). See `## Outcome`, "What §4 got
  right, and the one thing §7 got wrong", for the table and
  `ltx2_connector.h:157-161` for the shipped rule;
* attention and the feed-forward narrow at the points §4.3, §4.5, §4.6 and §4.8
  name;
* both residual adds round (§4.8);
* the output mask stays f32 (§4.7).

**The activation CONTAINERS stay `std::vector<float>` and hold bf16 VALUES, and
that is a recorded stop rather than an oversight.** Two reasons, both measured.
(a) The bytes are not here: the activations are `[1, 1024, 3840]` and
`[1, 1024, 1920]`, about 23 MB together, against the 4 GB the weights just gave
back. (b) Narrowing them would DELETE wave 1's two dtype gates, which is the
finding in §7.

### 5.4 The production path selects bf16 because the checkpoint is bf16

`ConnectorWeightSet::Ensure` and `RunConnectorFromFile` pass `kBF16` to
`Ltx2LoadConnectorWeights`, and `RunConnector` takes its compute dtype from the
weights. No render call site grows a parameter, and the arm follows the weights
the way upstream's does. The f32 loader arm stays: it is what the five existing
parity goldens are measured against, and deleting it would delete the reference.

## 6. Tests

### 6.1 RED FIRST, through a production entry point

The red is already specified by wave 1 and needs no new instrument. In
`tests/vllm/multimodal/test_ltx2_video.cpp`, the prompted-render case asserts

```cpp
CHECK(fox.trace.connector_video_not_bf16 > fox.trace.connector_video_values / 2);
```

on the f32 connector. This row flips both to `== 0` and captures the literal red
before touching `src/`. The counters are sampled inside `Ltx2VideoEngine::Generate`
on the buffer the DiT cross-attends over, so the case enters through the render
entry point and no case constructs the connector by hand to prove the class works.

### 6.2 The goldens come from upstream RUN IN BF16

`gen-ltx2-pipeline-goldens.py` section 10 gains a bf16 half that runs the SAME
five arms — Split, Interleaved, Float64, NoRegisters, GatedNoBias — with the
module `.to(torch.bfloat16)` on a bf16 input. Every parameter, tolerance and
failure case of the f32 half is preserved and one dtype changes. Each arm is
emitted twice, once unpatched and once with both attention callables pinned to
`SDPA_MATH` (§4.6), and the generator prints their distance.

The section also emits the §4 discriminators as goldens in their own right: the
narrowed RoPE tables, the small-row epsilon probe of §4.2 with its rejected
alternative, and the two rope roundings with theirs.

### 6.3 The value gate

Bit-exactness is attempted first and used where it holds. Where the reduction
order prevents it the bound is derived from the format — bf16 unit roundoff
`2^-8` relative to the golden's own magnitude — never fitted to the result, and
any case that needs it records its measured margin beside it.

### 6.4 The DTYPE gate, which is the one that cannot be faked

1. **The weights.** On one synthetic checkpoint, `Ltx2LoadConnectorWeights` is
   materialized at both dtypes and the byte counts compared: the bf16 arm must be
   EXACTLY half and its f32 map empty. Measured on the same input, so no number is
   quoted.
2. **The output, in BOTH directions, in the same case.** Every float the bf16 arm
   returns must satisfy `BF16ToF32(F32ToBF16(v)) == v`, and the f32 arm on the
   same fixture must fail it on most of the stream. An f32 path cannot pass the
   first; a fixture that had lost its sub-bf16 detail cannot pass the second. That
   is what makes the goldens meaningful, because a golden alone passes in both
   arms.
3. **At the engine level**, `connector_video_not_bf16` and
   `connector_audio_not_bf16` are 0 on the render path.

### 6.5 The mutations a fresh reviewer applies

* Delete the engine's `kBF16` argument to `Ltx2LoadConnectorWeights` and restore
  the f32 arm. §6.4(3) must red; the digests, absmax, frame bytes and determinism
  checks must NOT.
* Round the split rope like the interleaved one (§4.3). The Split golden must red
  and the Interleaved one must not.
* Use `bf16(1e-6)` for the norm epsilon. Only §6.2's small-row probe may red; the
  ordinary-magnitude goldens must stay green, which is what proves the probe is
  the instrument and the goldens are not.
* Leave the RoPE tables at f32 (§4.4). The value goldens must red.
* Delete the round-trip assertion in §6.4(2) and run the bf16 cases against the
  f32 arm. The value goldens alone must NOT be enough to red it.

## 7. Risks

* **`Ltx2TextConditioning`'s container is a TRAP, not a chore.** Wave 1 handed it
  here to be narrowed. Measured at this head, narrowing it turns wave 1's two
  strongest assertions into tautologies: `tower_video_not_bf16` is counted over
  that f32 container, and the unit gate asserts the round trip on the floats
  `Ltx2EncodePromptToConditioning` returns. A `std::vector<uint16_t>` passes both
  by construction and measures nothing. It buys about 12 MB. It narrows when the
  DiT's `Ltx2ModalityInput` seam narrows and a replacement gate exists, which is a
  later A24 wave; it stays owed here WITH this reason rather than being done for
  the sake of the handoff.
* **bf16 is lossy and this row makes the render less precise than it was.** That
  is the point: upstream's answer is the bf16 one. The f32 arm stays reachable and
  gated.
* **The reduction escape was expected to change, and it did NOT.** This risk was
  written predicting that the bf16 arm would accumulate the norm's sum in f32
  because upstream's `torch.mean` does. Measured, it is wrong: `torch.mean` is a
  blocked f32 reduction, and the f32 arm's existing f64 accumulator
  (`ltx2_connector.cpp`, the L3 precedent) reproduces it far better than a
  same-width sequential loop. **Both arms keep the f64 sum.** See `## Outcome`,
  "What §4 got right, and the one thing §7 got wrong".
* **No real-weights render.** This row gates the arithmetic against upstream
  executed on CPU at reduced dimensions. A full-render token gate on the shipped
  checkpoint needs a GPU lease and is owed.

## 8. Gates

```sh
# G1 — the refusal narrows; it must not vanish.
grep -n "connector: " src/vllm/model_executor/models/ltx2_connector.cpp

# G2 — the render path selects the bf16 arm, and the call sites are counted by
#      STRING because these line numbers have drifted three times.
grep -c 'RunConnector(' src/vllm/multimodal/ltx2_video.cpp
grep -c 'RunConnectorFromFile(' src/vllm/multimodal/ltx2_video.cpp

# G3 — the goldens are upstream's, regenerated at the pin.
python3 scripts/gen-ltx2-pipeline-goldens.py --ltx2 ~/_git/LTX-2 \
  --out tests/vllm/models/ltx2_pipeline_goldens.inc && git diff --stat

# G4 — the focused suites, both arms.
ctest --test-dir build -R 'ltx2_pipeline|ltx2_video' --output-on-failure

# G5 — the full gate.
scripts/agent-preflight.sh --staged
python3 scripts/check-pr-size.py --base origin/main --head HEAD
```

## 9. Evidence

1. The eight probes of §4, run against `fd4ded7f` with torch 2.11.0+cu130 on CPU.
   Recorded in `## Outcome` with their literal output.
2. The regenerated goldens, whose header carries the upstream revision the
   generator read from git.
3. The measured byte counts of §6.4(1), printed by the case itself.
4. The literal red of §6.1 and the literal green after.

## 10. Stop conditions

* A GPU lease is needed for anything here. Stop and report; do not take one.
* Upstream turns out to construct the connector at a dtype other than the
  pipeline's. Stop: the row's premise is wrong.
* A probe stops separating its hypotheses. Stop and rebuild the probe; do not
  emit a golden that cannot fail.

## Outcome

Everything below was measured on this branch. Where §4 predicted an answer and
the branch measured a different one, the branch wins and the difference is named.

### What §4 got right, and the one thing §7 got wrong

Every fact in §4 held: the norm's f32 epsilon, split rope's fused second term,
interleaved rope's three roundings, the RoPE tables' narrowing, the Linear's
single rounding, MATH's exactness, the register select and the GELU.

**§7 was wrong about the accumulator, and it was wrong in the direction that would
have cost values.** It said the bf16 arm would accumulate the mean square in f32
because upstream's `torch.mean` does. Measured against upstream at three widths,
with a sequential f32 loop and with the f32 arm's existing f64 one:

| width | sequential f32 sum | f64 sum | of |
|---|---|---|---|
| 24 | 0 | 0 | 96 000 |
| 120 | 2 | 0 | 480 000 |
| **3840** (shipped) | **11 000** | **24** | 15 360 000 |

`torch.mean` is a BLOCKED f32 reduction, so it is far more accurate than a
sequential f32 loop and is approximated much better by an exact f64 sum than by a
same-width one. The f64 escape stays on both arms, and the header now says this
was measured rather than inherited.

### The kernel that is not a function of its input, one level up from wave 1

`AttentionFunction.AUTOMATIC` resolves to
`SDPA[FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH>CUDNN_ATTENTION>OVERRIDEABLE]` and
FLASH serves the call on this CPU. Eight hypotheses over {round scores, round
probabilities, scale before/after the GEMM} were measured against it; the closest
differs on 128 of 384 bf16 words and a fully-materializing bf16 chain on 195.
`SDPBackend.MATH` is bit-equal to an f32-accumulated attention with no
intermediate rounding — 0 of 384 against both an f32 and an f64 reference — so the
goldens are emitted against MATH and the distance to the unpatched module is
reported. The generator prints it per arm:

```
  bf16 connector Split:       MATH vs AUTOMATIC differs on 57/384 words, max|diff| 0.015625  (max|golden| 2.6875)
  bf16 connector Interleaved: MATH vs AUTOMATIC differs on 38/384 words, max|diff| 0.00842285 (max|golden| 2.9375)
  bf16 connector Float64:     MATH vs AUTOMATIC differs on 19/384 words, max|diff| 0.0078125  (max|golden| 2.26562)
  bf16 connector NoRegisters: MATH vs AUTOMATIC differs on 12/384 words, max|diff| 0.0078125  (max|golden| 2.28125)
  bf16 connector GatedNoBias: MATH vs AUTOMATIC differs on 61/384 words, max|diff| 0.0117188 (max|golden| 2.78125)
```

The f32 arm has the same problem and has always lived with it: at f32 the two
backends differ by 4.77e-7 across the whole connector, under `kRoundOff`.

### §6.3 planned a tolerance and the tree does not need one

**The port is BIT-EXACT to the MATH oracle on all five arms — max|diff| 0.0.** The
format-derived bound §6.3 planned is 0.0177 to 0.0229 on these arms, and keeping
it would have been a mute switch rather than a safety margin. Measured, under that
bound:

* mis-rounding SPLIT rope reds **1 of the 4** split arms; under bit-exactness, 4 of 4;
* leaving the RoPE tables at f32 moves 3 of 5 arms by LESS than the bound;
* running the arithmetic at f32 with bf16 weights moves **all 5** by less than it.

The bound is still computed and printed beside the measured difference so a reader
can see how much room the format would have allowed. The gate is `worst == 0.0`
plus `to_unpatched == kernel_gap`, which follows from bit-exactness and breaks if
either side moves.

### The dtype gate, which is the row

Unit level, both directions on one fixture, five arms:

```
bf16 arm returns 0 of 384 values wider than bf16; the f32 arm, same config and
same input, returns 384 of 384.
```

Engine level, on the render path, before and after:

```
RED  (f32 connector, the tree at 8e582a5f9 with the assertion flipped)
  test_ltx2_video.cpp:7012: ERROR: CHECK( fox.trace.connector_video_not_bf16 == 0 )
    values: CHECK( 16384 == 0 )
    logged: tower output wider than bf16: video 0 of 16384, audio 0 of 8192
            connector (f32, wave 2) wider than bf16: video 16384 of 16384, audio 8191 of 8192
  [doctest] assertions: 67 | 65 passed | 2 failed |

GREEN (this branch)
  [doctest] test cases:  1 |  1 passed | 0 failed | 112 skipped
  [doctest] assertions: 67 | 67 passed | 0 failed |
```

**The 65 assertions that never moved are the argument for A24.** The digests, the
absmax, the prompt dependence, the frame bytes and the determinism check all
passed while the conditioning the DiT cross-attends over was twice as wide as
upstream's.

### The five mutations, with their literal results

The tree was restored with `git checkout --` after each and re-verified green.

| mutation | result |
|---|---|
| **A. delete the production call site** — revert `ConnectorWeightSet::Ensure` to the f32 loader arm | `CHECK( 16384 == 0 )` and `CHECK( 8191 == 0 )`. `assertions: 67 \| 65 passed \| 2 failed`. Exactly the two dtype assertions; nothing else on the render path moved. This is the reachability proof. |
| **B. round split rope like the interleaved arm** | reds Split, Float64, NoRegisters and GatedNoBias — all four split-rope arms — and leaves Interleaved green. `assertions: 409 \| 403 passed \| 6 failed`. |
| **C. use `bf16(1e-6)` for the norm epsilon** | reds ONLY the epsilon probe, on 2 assertions: `worst == 0.0` and `to_rejected > 0.0`. `assertions: 722 \| 720 passed \| 2 failed`. All ten arm goldens, f32 and bf16, stay green — which is the claim that the arm goldens cannot hold the epsilon and the small-row probe can. |
| **D. leave the RoPE tables at f32** | reds all five arms. `assertions: 722 \| 714 passed \| 8 failed`. Three of the five moved by less than the format bound, which is how this mutation proved the bound had to go. |
| **E. bf16 weights, f32 arithmetic** | reds all five arms on the value golden AND on `narrow_wider == 0`, `assertions: 15 failed`. Under the ORIGINAL bound only `narrow_wider` would have fired. At the ENGINE level there is no golden at all, so the counter is the only instrument there — which is what A measured. |

### Why `Ltx2TextConditioning` is still owed, measured rather than argued

Wave 1 handed this row that container to narrow. Narrowing it turns wave 1's two
strongest assertions into tautologies: `tower_video_not_bf16` is counted over that
f32 buffer, and wave 1's unit gate asserts the bf16 round trip on the floats
`Ltx2EncodePromptToConditioning` returns. A `std::vector<uint16_t>` satisfies both
by construction. It buys about 12 MB at the shipped widths, against the ~4 GB the
weights arm returns. It narrows when the DiT's `Ltx2ModalityInput` seam narrows
and a replacement gate exists.

### One record repaired in flight

`ltx2_video.cpp`'s READER ANCHORS list went stale because this row's comments moved
the lines under it, and `test_ltx2_video`'s own anchor gate caught it and printed
the replacement. Recorded because it is the third time this file's line anchors
have drifted inside one pull request, and because it is why §3 above cites a
`grep -c` on a call string instead of a line number.

### What the fresh review found, and what the repairs measured

Three findings were required and four more were applied. All of them are in the
branch; none of them was re-litigated.

**F1 was a correctness defect on the shipped branch, not a gate gap.**
`Ltx2ConnectorForward` narrowed the caller's stream inside its `else` only, so
the `num_learnable_registers > 0` branch — the SHIPPED one, because the
checkpoint declares 128 — handed the first transformer block whatever width the
caller carried. The substitution replaces only the padded rows and copies the
kept ones verbatim, so the comment claiming both sides of the select were already
bf16 was true of the registers and false of everything else. It is reachable:
`Ltx2VideoEngine::Load` and `GenerateAudioOnly` pass `RunConnectorFromFile` the
output of `ReadF32File`, an arbitrary user file. The tower path escaped only
because wave 1 makes the tower output bf16-valued, which is an invariant enforced
in a different component.

No case here could see it, because every bf16 golden is emitted from
`hidden.to(bf16)` and a missing narrowing over a bf16-valued input is an
identity. The new case perturbs that fixture by a QUARTER of a bf16 ulp —
`x * (1 + 2^-10)` against the format's own `2^-8` — and asserts the forward
answers exactly as it does on the pre-narrowed stream:

| arm | before the repair | after |
|---|---|---|
| Split (4 registers) | `CHECK( 0.015625 == 0 )` | `0` |
| NoRegisters | `0` | `0` |

The `NoRegisters` arm is in the case so the pair says WHICH branch moved. The
narrowing is an identity over the substituted register values, so the four
register arms stay bit-exact against upstream after it.

**F2: the q/k RMSNorm's epsilon was reached and ungated.** Mutation C in the
table above holds `Ltx2ConnectorRmsNormRows`, the connector's WEIGHTLESS residual
norm. The q/k norms are `torch.nn.RMSNorm(inner_dim, eps=norm_eps)`
(attention.py:505-506), they take the same constant in the same forward, and they
reach it through a DIFFERENT function — `Ltx2RmsNormRows` in `ltx2.cpp`. Narrowing
that epsilon to `bf16(1e-6)` at `kBF16` left `test_ltx2_pipeline` at
`4018 | 4018 passed` and `test_ltx2_video` at `4951 | 4951 passed`. The control,
setting it to `1.0`, reds 11 — ten across all five connector arms plus the new
probe — so the site was live and reached, and the arm goldens simply could not
resolve it.

The arithmetic did NOT change; the probe did. `section_connector_bf16`'s small-row
probe grew a weighted arm on the same 2^-13 rows with a bf16 gain and the same
refuse-if-not-separating guard. Measured by the generator at width 24:

| row scale | upstream vs f32-eps | upstream vs bf16-eps | of |
|---|---|---|---|
| 2^-13 | **0** | 1831 | 12 288 |
| 2^-8 | **0** | 62 | 12 288 |
| 2^-2 | **0** | 0 | 12 288 |

Re-running the narrowing mutation with the probe in place: `test_ltx2_pipeline`
goes from `4018 | 4018 passed | 0 failed` to
`4018 | 4016 passed | 2 failed`, and the two are the new probe's own —
`CHECK( 0.00195312 == 0 )` and `CHECK( 0 > 0 )` — with every other assertion,
`test_ltx2_video` included, still green. That is the literal before and after:
without these two assertions the mutation was invisible.

`Ltx2RmsNormRows` is declared in `ltx2.h` for this, on the same argument
`ltx2_connector.h` already makes for `Ltx2ConnectorRmsNormRows`: it is
`Ltx2Attention`'s own norm, called twice per block, and no forward fixture
produces rows near 2^-13.

**F3 and F4 were spec repairs.** §5.3 and §7 still stated the f32 accumulator this
branch measured and rejected; both now say f64 and point at this Outcome. §4.1
recorded round-then-multiply as "NOT separable" from the fused form, and executing
it parts upstream from it on 12 647 of 49 152 values at ordinary magnitude and
3 166 of 12 288 at the probe's rows. It is retracted in place, and the alternative
is now a SECOND rejected hypothesis in the weighted probe rather than a sentence
saying it could not have one.

**F5, F6, F8 and F9.** Nine of the twelve emitted
`kLtx2ConnBf16*{Registers,Replaced,ZeroedMask}Golden` arrays were read by nothing;
the generator now emits the trio for `Split` alone, the arm whose isolation case
reads all three, and the other three register arms keep the end-to-end forward
golden that already subsumes the substitution. `Ltx2VaeWeights::Has()` inspected
only the f32 map while `Get` and `Count` were arm-aware, so it answered NO to
every tensor a bf16 bag holds; it is arm-aware now, before the VAE decoder wave
reaches it. `Ltx2LoadConnectorWeights`' `compute_dtype` lost its `kF32` default,
so no future call site gets the WIDER arm by silence. And `ltx2.cpp` cited
`rope.py:73` for `output = split_input * cos`, which is at `:71`.

## Owed

* **The other six A24 components**, none of which this row touches: the video VAE
  decoder (`ltx2_video_vae.h:47-54`), the video VAE encoder
  (`ltx2_video_vae_encoder.h:52-57`), the video VAE device kernels
  (`ltx2_video_vae_kernels.h:44-51`), the tiled-decode buffer
  (`ltx2_tiling.h:88-94`), the latent upsampler (`ltx2_upsampler.h:66-70`) and the
  duration head (`ltx2_duration_head.h:55-58`).
* **`Ltx2TextConditioning`'s f32 container**, still, with §7's reason. It narrows
  with the `Ltx2ModalityInput` seam and a replacement for wave 1's counters.
* **The connector's activation containers** (§5.3): about 23 MB at the shipped
  widths, against the 4 GB this row returns.
* **The FP8 and NVFP4 arms**, which are A22.
* **A real-weights render in bf16 against upstream**, which needs a GPU lease.

## Now

`DONE`. The spec commit precedes the implementation commit, which is the commit
order that proves it came first.
