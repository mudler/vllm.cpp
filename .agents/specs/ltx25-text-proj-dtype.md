# LTX25-TEXT-PROJ-DTYPE — the caption projections assume NVFP4, so the bf16 tower cannot load

Row: `LTX25-TEXT-PROJ-DTYPE`, under the LTX-2.5 campaign
([`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#2140](https://github.com/mudler/vllm.cpp/issues/2140).
Oracle: Lightricks `LTX-2` at pin `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
([`ltx-2.md`](../oracles/ltx-2.md)). vLLM registers nothing LTX at the parity
pin, and vLLM-Omni's `ltx2` recipes stop at generation 2.3, so the model
author's own runtime is the reference for this rule.

## Now

`ACTIVE`. One function changes: `LoadProjection` in
`src/vllm/model_executor/models/ltx2_loader.cpp`. Nothing else on the
LTX-2.5 path moves.

## 0. Scope

**In scope.** `LoadProjection` resolves the caption projection's storage format
from the file instead of assuming torchao-NVFP4, so a BF16 caption projection
loads at its true width. The NVFP4 arm keeps every refusal and every byte it has
today. A load-time case on a synthetic BF16-projection fixture gates the new arm
without a lease.

**Also in scope, because the evidence needs it.**
`scripts/probe_ltx2_text_encoder_load.cpp`, a probe in the shape of the two that
`scripts/` already carries: no CMake target, and its compile line recorded in its
own header so a reviewer can re-run it. A synthetic fixture cannot prove that the
real bytes of either shipped encoder resolve, and this row's whole subject is
that the two files disagree about their format.

**Not in scope.** The Gemma tower itself, which already has both arms
(`TowerModule` in `ltx2_text_encoder.cpp` takes a `BF16` module directly and a
`U8`-plus-marker module through the dequantizer). The comparison gate of
`LTX25-ORACLE-ABSOLUTE` and its branch. Any change to
`Ltx2WidenTextProjectionsToF32`; §3 establishes by reading why it needs none.
Any render number: §6 says what is owed and why.

## 1. The defect, measured

`rc` job `001c36e9-76b1-432c-9536-2d24c0e613d0` on `dgx:gpu0`, 2026-08-27, on
`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` sha256
`ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1`, recorded in
[#2140](https://github.com/mudler/vllm.cpp/issues/2140). The tower loads in
34.815 s and then the load refuses:

```text
'text_embedding_projection.video_aggregate_embed.weight' unpacks to in_features
376320 but the Gemma geometry gives 188160
```

`LoadProjection` computes `proj.in_features = w->shape[1] * 2` unconditionally,
with the comment "NVFP4 packs TWO values per byte". On a BF16 checkpoint the
stored width is already the logical one, so the doubling corrupts a correct
number and the geometry check fires on the loader's own arithmetic. It then
requires `.weight_scale` and `.weight_scale_2` and dequantizes, neither of which
a BF16 file has.

**The two files' headers, read here rather than reasoned about.** Both live on
the shared NAS; only the 8-byte length prefix and the JSON header were read, not
the tensor bytes:

| Tensor | bf16 file | nvfp4-torchao file |
|---|---|---|
| `model.norm.weight` | `BF16 [3840]` | `BF16 [3840]` |
| `...video_aggregate_embed.weight` | `BF16 [4096, 188160]` | `U8 [4096, 94080]` |
| `...video_aggregate_embed.weight_scale` | absent | `F8_E4M3 [1024, 47040]` |
| `...video_aggregate_embed.weight_scale_2` | absent | `F32 []` |
| `...video_aggregate_embed.torchao_nvfp4` | absent | `U8 [240]` |
| `...video_aggregate_embed.bias` | `BF16 [4096]` | `BF16 [4096]` |
| `.weight_scale` tensors in the file | **0** | **334** |
| `torchao_nvfp4` markers in the file | **0** | **334** |

188160 = 3840 * (48 + 1), which is the geometry the refusal quotes. The bf16
file carries 681 BF16 tensors and 5 U8 ones, and the 5 are the tokenizer and
asset pack, not weights. So the bf16 arm is not a partially quantized file: it
has no quantized module at all, and `quantized_modules` is legitimately empty
for it.

**The irony that says what to fix.** Three lines below the doubling, the same
function resolves the NVFP4 *producer* from the tensor rather than hard-coding
`kTorchao`, with a comment that hard-coding "would make the projections the one
NVFP4 path in the loader that cannot notice a producer change". The subtle
assumption is guarded and the gross one is baked in. This row brings the dtype
branch up to the standard the producer resolution already sets.

## 2. What upstream's rule actually is

Upstream does not guess a format from a dtype, and it does not carry a
per-module flag either. It **discovers** quantized layers from the safetensors
header, and the rule is one function:
`_discover_nvfp4_layers`, `packages/ltx-core/src/ltx_core/quantization/nvfp4/prequant.py:30-50`
at pin `fd4ded7f`. Three clauses, in upstream's own order:

1. A layer is a candidate when it has **both** `.weight_scale` and
   `.weight_scale_2` (`:35-36`, `:42-43`).
2. Having **exactly one** of the pair is an error, not a fallback:
   "expected both or neither (NVFP4 checkpoints pair them 1:1:1)" (`:37-41`).
3. A candidate is NVFP4 only when the dtype triple is
   `U8` / `F8_E4M3` / `F32` (`:47-48`).

Everything the discovery does not select stays the plain `nn.Linear` upstream
built for it. For these two modules that is
`torch.nn.Linear(flat_dim, video_inner_dim, bias=True)` and its audio sibling,
`packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py:206-208`,
where `flat_dim = embedding_dim * (num_hidden_layers + 1)` (`:181-183`) — the
same 188160. A plain Linear stores one value per element, so **for the
unquantized arm the stored width IS the logical width**, and that is the port's
authority for dropping the `* 2` rather than a local heuristic.

Upstream's `_discover_prequant` also raises when a checkpoint asked for the
prequant path carries no NVFP4 layer at all (`:90-93`); that is a *pipeline
selection* refusal on the transformer, and its module swap is scoped to
`isinstance(model, LTXModel)` (`:204`), so it never reaches the text encoder.
This port must therefore accept a text encoder with zero quantized modules,
which upstream reaches by never running the discovery there.

**One place this port is deliberately stricter than upstream, and why.** Clause
3 makes upstream `continue` — a `U8` weight with a mismatched scale dtype is
simply not NVFP4, and it then fails later inside `load_state_dict`. This loader
refuses by name instead, in the shape `TowerModule` already uses for the tower:
"neither the BF16 form nor the torchao-NVFP4 form this loader understands". It
accepts nothing upstream rejects; it only names the refusal earlier. The
alternative is reading a packed byte stream as bf16 values, which produces a
plausible tensor of the right shape and the wrong numbers.

## 3. The change

In `LoadProjection`, after the rank-2 check and before any width arithmetic:

- Look up `.weight_scale` and `.weight_scale_2`. If exactly one is present,
  fail naming the module and both suffixes, mirroring `prequant.py:37-41`.
- **Quantized arm** — both present: keep today's behaviour byte for byte.
  `in_features = w->shape[1] * 2`, the same geometry refusal, the same producer
  resolution, the same `Ltx2DequantNvfp4ToBf16` call. The `U8` requirement moves
  from implicit to stated, because a non-`U8` weight beside a scale pair is
  clause 3's case.
- **Plain arm** — neither present: require `BF16`, take
  `in_features = w->shape[1]`, check `nbytes` against
  `out_features * in_features * 2`, and `memcpy` the rows into `weight_bf16`.
  A non-`BF16` weight with no scales refuses by name.

The geometry refusal's text becomes arm-specific. Today it always says "Reading
the STORED U8 width as logical is what halves it", which is the wrong diagnosis
on a bf16 file — it was the sentence that sent #2140's first reader to
`feature_extractor.py`. Each arm now states the width it read and how it read
it.

The bias path is untouched: it is already `BF16` on both arms, which the header
table above measures, and it is the split
`ltx2_text_encoder.h` names as the one a loader silently half-does.

**`Ltx2WidenTextProjectionsToF32` needs no change, established by reading.** Its
`widen` lambda copies `out_features` and `in_features` straight off the
`Ltx2TextProjection` and widens `weight_bf16` and `bias_bf16` elementwise. It
never reads a shape from the file and never sees a dtype. Both arms hand it the
same already-dequantized bf16 buffer at the same logical width, so it is correct
for the plain arm exactly when `LoadProjection` is. It is coupled to the fix in
the sense that a wrong `in_features` propagates through it, which is why the
bf16 case asserts on the widened result as well as on the checkpoint.

The file-level validation loop in `Ltx2LoadTextEncoderFromSafetensors` also
doubles a stored width, and that one is correct: it is keyed on the presence of
a `torchao_nvfp4` marker, so on the bf16 file its body never runs. No change.

## 4. Risks

- **Breaking the NVFP4 arm while fixing bf16.** It is the shipped default and it
  is gated. Mitigated by keeping the quantized branch's statements in their
  existing order and by running its existing cases before and after (§5). If the
  bf16 arm cannot be made to work without changing NVFP4 behaviour, this row
  stops and reports rather than choosing.
- **A silent wrong-width read.** A bf16 weight whose stored width is genuinely
  half the geometry would now pass the `* 1` arm's arithmetic and fail the
  geometry check, which is the correct outcome; the case is gated.
- **A mixed file.** A checkpoint with one projection quantized and the other not
  is resolved per module, because the discovery is per module upstream too. No
  such file is known to ship.
- **An unreachable fix.** A loader arm that no production path enters is the
  failure `.agents/reachability.md` names. Gated by the mutation in §5.

## 5. Gates and evidence

1. **Red first.** A synthetic bf16 text-encoder fixture, built by the same rules
   as the existing torchao one at reduced dimensions, loaded through
   `Ltx2LoadTextEncoderFromSafetensors`. Red before the change with the
   in_features doubling in the message; green after.
2. **The NVFP4 arm, before and after.** `test_ltx2_loader` in full, and the
   `--test-case` subset that names the torchao text encoder, run at the base
   commit and at the head.
3. **Reachability.** Delete the production `LoadProjection` call site in a
   scratch copy and rerun the focused gate. It must go RED. A gate that stays
   green without the call site measures a class, not a capability.
4. **Refusal cases.** Exactly one of the scale pair present; a `U8` weight with
   no scales; a `F32` weight with no scales. Each refuses naming the module.
5. **`scripts/agent-preflight.sh`** green, including `--staged`.

## 5b. What was measured

Every number below was produced on this branch, on this host (`mudler-ubuntu-box`,
x86-64, CPU-only Release build, no GPU and therefore no lease: nothing here
touches a device).

**The checkpoints, hashed before they were read.** Both agree with the records
that pin them, so neither is the re-quantized-in-place case #1723 records:

| File | sha256 measured here | Agrees with |
|---|---|---|
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1` | `tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json` and `docs/USAGE.md` |
| `gemma4-12b-with-proj-nvfp4-torchao.safetensors` | `12132b7157925332d2b21de9fc6f507c14f4f0cbc7081484d1968ebf8a19b4bf` | `docs/USAGE.md` |

**Red before, green after, on the synthetic fixture.** The four new cases fail
at the base source with `unpacks to in_features 1024 but the Gemma geometry
gives 512` — the same factor of two, at the fixture's reduced dimensions, that
the shipped file shows as 376320 against 188160. After the change: 4 cases, 42
assertions, all passing.

**Red before, green after, on the SHIPPED bf16 checkpoint.** Through
`scripts/probe_ltx2_text_encoder_load.cpp`, which calls the same
`Ltx2LoadTextEncoderFromSafetensors` the engine calls. At the base source:

```text
REFUSED: ltx2 loader: 'text_embedding_projection.video_aggregate_embed.weight'
unpacks to in_features 376320 but the Gemma geometry gives 188160
```

which is #2140's message, character for character. At this head, the same file:

```text
gemma_hidden_size       = 3840
gemma_num_hidden_layers = 48
geometry hidden*(L+1)   = 188160
video  out=4096 in=188160 weights=770703360 bias=4096
audio  out=2048 in=188160 weights=385351680 bias=2048
quantized_modules       = 0
tokenizer_json bytes    = 32169626, has_config=1
video.weight[0..3]      = 0.01440430 -0.00057220 -0.00286865 0.00201416
widened video out=4096 in=188160 weights=770703360 bias=4096
VmHWM kB                = 9100848
OK
```

**The NVFP4 arm is unchanged, on the shipped torchao checkpoint too.** The same
probe on `gemma4-12b-with-proj-nvfp4-torchao.safetensors` reports the identical
geometry and the identical projection widths, with `quantized_modules = 334` and
`has_config=0`, and its `video.weight[0..3]` reads
`0.01397705 -0.00000000 -0.00233459 0.00233459` against the bf16 file's
`0.01440430 -0.00057220 -0.00286865 0.00201416`. Those are two encodings of one
tensor, and the agreement is a cross-arm sanity signal rather than a gate:
nothing here asserts a tolerance between the arms, because a 4-bit encoding of
the second element legitimately reads as zero.

**The NVFP4 arm's own cases, before and after.** `test_ltx2_loader` filtered to
`*torchao*,*NVFP4*,*nvfp4*,*require_config*` is **11 cases / 19934 assertions,
0 failed** at the base source and **11 cases / 19934 assertions, 0 failed** at
this head — the same two numbers, not merely both green. The full binary is 41
cases / 64246 assertions, 0 failed, and all 13 `ctest -R ltx2` targets pass.

**Four mutations, each restored byte-for-byte afterwards** (verified with
`sha256sum -c` over the three touched files):

| Mutation | Focused gate |
|---|---|
| Delete the production `LoadProjection` call site in `Ltx2LoadTextEncoderFromSafetensors` | **RED**, 4/4 cases, 18 assertions failed |
| Restore the `* 2` on the plain arm only (the #2140 defect) | **RED**, 2/4 cases |
| Disarm the exactly-one-of-the-scale-pair refusal | **RED** |
| Disarm the plain arm's `BF16` dtype refusal | **RED** |

The first is the reachability case: without the call site the gate measures a
class rather than a capability, and it does not stay green.

**Re-measured after `origin/main` was merged in.** `main` moved 10 commits
during this row, and a merge can falsify a claim made before it without touching
a line of the claim. So every number above was taken again at the merge commit
rather than carried forward: `test_ltx2_loader` is 41 cases / 64246 assertions /
0 failed, its NVFP4 subset is 11 cases / 19934 assertions / 0 failed, all 13
`ctest -R ltx2` targets pass, and the probe on the shipped bf16 file prints the
same `video out=4096 in=188160`, `quantized_modules = 0`, `OK`.

**What this does NOT prove.** No render ran, so nothing here says the bf16 arm
produces the right video — only that its weights arrive at the right width with
the right bytes. §6 keeps that owed.

## 6. Owed

- **The end-to-end bf16 render.** §5b proves the shipped bf16 text encoder now
  LOADS, on its own bytes, through the function the engine calls. It does not
  prove the arm renders: that needs the DiT and both VAEs on a device, which
  needs a `dgx:gpu0` lease. No lease was taken by this row — `dgx:gpu0` was held
  by an unrelated job with another hold queued behind it, and displacing either
  to prove a load that a CPU-only probe already proves would be the wrong trade.
  The render stays owed here and is what
  [#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s comparison will
  exercise.
- **The comparison reading itself** stays with `LTX25-ORACLE-ABSOLUTE`. This row
  unblocks it and does not take it.

## 7. Stop conditions

- Fixing bf16 requires changing NVFP4 behaviour: stop, report `NEEDS_DECISION`.
- A checkpoint sha256 disagrees with `ltx2_oracle_manifest.json`: stop.
- The fleet is unreachable, or `dgx:gpu0` is held: land the unit-level fix and
  record the render as owed. Do not claim an end-to-end result that was not run.
