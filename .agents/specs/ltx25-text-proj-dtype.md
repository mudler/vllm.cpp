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

## 6. Owed

- **The end-to-end bf16 render.** The only exercise that proves the whole arm is
  a GPU render, and it needs a `dgx:gpu0` lease and the four bf16 checkpoints
  the `ltx2_oracle` manifest pins. Recorded as owed on
  [#2140](https://github.com/mudler/vllm.cpp/issues/2140) unless this row takes a
  lease and records the result here. A unit gate proves the width arithmetic; it
  does not prove that the render is right.
- **The comparison reading itself** stays with `LTX25-ORACLE-ABSOLUTE`. This row
  unblocks it and does not take it.

## 7. Stop conditions

- Fixing bf16 requires changing NVFP4 behaviour: stop, report `NEEDS_DECISION`.
- A checkpoint sha256 disagrees with `ltx2_oracle_manifest.json`: stop.
- The fleet is unreachable, or `dgx:gpu0` is held: land the unit-level fix and
  record the render as owed. Do not claim an end-to-end result that was not run.
