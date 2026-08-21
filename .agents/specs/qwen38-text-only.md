# Qwen3.5/3.8 text-only arms: `Qwen3_5MoeForCausalLM`, `Qwen3_5ForCausalLM`

**Rows:** `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm`,
`MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` (both NEW beyond-pin rows; this row
adds them to [`model-matrix.md`](../model-matrix.md) with the inventory counts
bumped, mirroring the `MuseGlimmer`/`KimiK3` beyond-pin precedent)
**Issue:** [#490](https://github.com/mudler/vllm.cpp/issues/490)
**Also closes:** [#627](https://github.com/mudler/vllm.cpp/issues/627) — the
pre-existing misaligned-load UB this row's new test was the first gate to reach
(see `## Outcome` → "The safetensors alignment class").
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Load text-only checkpoints of the Qwen3.5-family GDN-hybrid backbone we already
run — the arms upstream calls `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`.
`Qwen/Qwen3.8-2.4T-A95B` is the motivating checkpoint; it is the same
architecture at larger scale, not a new one.

In scope:

- register the two text-only architecture strings against the existing dense and
  MoE factories;
- accept both the VL-prefixed (`model.language_model.`) and clean (`model.`)
  weight namespaces in the Qwen3.5 dense and MoE loaders;
- resolve a **flat** (non-nested, no `vision_config`) text config through the
  existing path;
- prove the 27B / 35B / Coder gates stay byte-identical.

Out of scope: any speed claim, any GGUF arm for 3.8, the vision tower
(a text-only checkpoint has none), MTP weights for 3.8, advancing the parity pin,
the **bf16 / 3-D-stacked MoE routed-expert arm** (owed, and refused by name
here — see [What this row does NOT make loadable](#what-this-row-does-not-make-loadable)),
and **any support claim for the 2.4T checkpoint itself**, which this hardware
cannot execute (see Gates).

## Why this is not a new port

`config.json` for `Qwen/Qwen3.8-2.4T-A95B` declares `Qwen3_5MoeForCausalLM` /
`model_type: qwen3_5_moe_text`. Against Qwen3.6-35B-A3B, which we run token-exact
315/315, every structural knob is identical — `head_dim` 256,
`linear_key/value_head_dim` 128, `linear_num_key_heads` 16,
`full_attention_interval` 4, `attn_output_gate` true, `partial_rotary_factor`
0.25, `rope_theta` 1e7, `mtp_num_hidden_layers` 1, `linear_conv_kernel_dim` 4,
and `vocab_size` 248320 (the same tokenizer). The differences are scale only:
hidden 2048->8192, layers 40->92, attention heads 16->64, KV heads 2->4, linear
V-heads 32->128, experts 256->512, top-k 8->10, moe/shared intermediate
512->2048. All of these are read from config, not hardcoded
(`qwen3_5_common.cpp:40-47`; the only expert constraint is `num_experts > 0` at
`qwen3_5_weights.cpp:618`).

The 3.8 config also carries `output_gate_type: "swish"`, which normalizes to
silu. That key is handled by its own row (issue #489) and is not re-litigated
here.

## Upstream chain

| Upstream anchor | Contract to mirror |
|---|---|
| upstream `vllm/model_executor/models/registry.py:202-203` @ `ad5d29db7` | `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM` are registered text-only arms of the same `qwen3_5` module. |
| upstream `vllm/model_executor/models/qwen3_5.py:439-449` @ `ad5d29db7` | `Qwen3_5ForCausalLM` is the shared base unchanged; `Qwen3_5MoeForCausalLM` is that base plus the MoE hyperparameters — no separate backbone. |
| upstream `vllm/model_executor/models/qwen3_5.py:296-300` @ `ad5d29db7` | `WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})` — the canonical namespace is `model.`, and the VL-prefixed form is accepted and rewritten. |

**Ahead-of-pin, stated as such.** Our parity pin is `555967922`, whose registry
carries only the `ForConditionalGeneration` entries. The text-only arms arrived
upstream in **PR #50210 / `ad5d29db7`**, which is post-pin. This row is a
deliberate forward port of one upstream PR, not a mirror of the pin, and it does
not advance the pin or reconcile anything else in that range. That is visible
debt argued here and in the commit, not a silent divergence.

## Design

Weight-name evidence, read from the published indices of both checkpoints:

| | Qwen3.6-35B-A3B (published) | Qwen3.8-2.4T-A95B (published) |
|---|---|---|
| embed | `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| layer | `model.language_model.layers.0.linear_attn.*` | `model.layers.0.linear_attn.*` |
| experts | `...mlp.experts.gate_up_proj` + `.down_proj` (3D STACKED, 41x) | `...mlp.experts.gate_up_proj` + `.down_proj` (3D STACKED, 93x) |
| shared | `...mlp.shared_expert_gate.weight` | `...mlp.shared_expert_gate.weight` |
| head | `lm_head.weight` | `lm_head.weight` |
| quant scales | **NONE** — 0 `weight_scale`, 0 `input_scale` | **NONE** — 0 `weight_scale`, 0 `input_scale` |

The **backbone** names are identical modulo the prefix, so the namespace
decision is the whole of what this row changes in the loader. **It is not the
whole of what either checkpoint needs**, and an earlier revision of this spec —
and of the two commits below it — wrongly said it was. See
[What this row does NOT make loadable](#what-this-row-does-not-make-loadable).

1. **One prefix decision, resolved once.** The Qwen3.5 loaders currently
   concatenate the literal `model.language_model.` in 4 places
   (`qwen3_5_weights.cpp:560,632,633,659`) and 3 more in
   `qwen3_5_dense_weights.cpp`. Replace the literal with a single resolved
   backbone prefix, chosen once per checkpoint by probing which namespace the
   shard index actually contains, then used everywhere. Mirrors upstream's single
   `WeightsMapper` rather than scattering a fallback into each lookup — a
   per-lookup fallback would let a checkpoint load half from one namespace and
   half from the other and still appear to succeed.
2. **Registration is additive.** Two `REGISTER_VLLM_MODEL` entries pointing at
   the existing dense and MoE factories. No factory, forward, or KV-cache change:
   `ModelRegistry::Resolve` is exact-match with no aliasing
   (`model_registry.cpp:217-231`), so the strings must be present literally.
3. **Config resolution already works.** `ResolveTextConfig` falls through to the
   top-level document when there is no `text_config`
   (`hf_config.cpp:113-122`), and `qwen3_5_moe_text` is already in
   `IsQwen35Family` (`:128-132`), so the `partial_rotary_factor` 0.25 default
   applies to a flat config. MRoPE is mm-path-only and every text caller passes
   `nullptr` (`qwen3_5.cpp:7540`), so a config without `mrope_section` is
   unaffected. Both facts get a test rather than an assumption.

## What this row does NOT make loadable

Corrected 2026-08-12 after an independent review returned FAIL on records
honesty. The registration and the namespace resolution are sound; the claim
built on top of them was not.

**The MoE arm cannot read a published Qwen3.5-family MoE checkpoint, in either
namespace.** `LoadQwen3_5Moe` routes every routed expert through
`LoadMoeExpertsInto` (`qwen3_5_weights.cpp:519-530`) into `LoadNvfp4Raw`
(`:433-462`), which hard-requires per-expert `experts.<e>.<proj>.weight` = `U8`,
`.weight_scale` = `F8_E4M3` and `.weight_scale_2`. There is **no stacked branch
and no bf16 branch** — unlike `gemma4_weights.cpp:326`, which dispatches between
layouts. Against that, the published indices (read live 2026-08-12):

- `Qwen/Qwen3.8-2.4T-A95B`: 1609 tensors, 93x `mlp.experts.gate_up_proj` +
  93x `.down_proj` (3-D stacked, 92 backbone layers + 1 MTP), **zero** names
  matching `weight_scale` or `input_scale`, `lm_head.weight` alone.
- `Qwen/Qwen3.6-35B-A3B`: 1045 tensors, the same stacked spelling under
  `model.language_model.`, **zero** `weight_scale`.

So the 2.4T load would die at `w.lm_head_fp4 = LoadNvfp4Raw(get, "lm_head")`
(`:679`) before the experts are even reached, and would die again at the FP8
attention, the routed experts and the shared expert. Our gated 35B row reads the
REQUANTIZED `nvidia/Qwen3.6-35B-A3B-NVFP4`; this loader **has never read a
published Qwen bf16 MoE repo**.

**The dense/MoE asymmetry is real and must not be flattened.**
`LoadQwen3_5Dense` DOES route BF16 vs FP8 vs NVFP4 per projection by tensor
presence (`qwen3_5_dense_weights.cpp:355-361,473-504`, and
`LoadDenseLmHead`/`LoadLmHeadAnyDtype` at `:215-233,515-547`), so the DENSE
text-only arm may genuinely load a flat bf16 checkpoint. Only the MoE arm
cannot. Any statement about "the text-only arms" that does not make that
distinction is wrong.

**What is therefore OWED, named:** the **bf16 / 3-D-stacked MoE routed-expert
arm** (plus the bf16 shared expert, the bf16 FP8-less attention tower, and the
bf16 `lm_head`, all on the MoE path). That is a real port with its own spec,
RED-first test and NVFP4 inertness proof — it is not this row. Until it exists,
this row ships a **REFUSAL that names the missing piece**
(`CheckMoeExpertLayoutSupported`, `qwen3_5_weights.cpp`), because AGENTS.md
requires an unimplemented arm be refused by name rather than discovered later,
and this spec's own stop conditions said the same.

**Consequence for the run gate.** "It closes when a text-only
`Qwen3_5[Moe]ForCausalLM` checkpoint that fits GB10 appears" is FALSE for the
MoE arm: a fitting *published* (bf16/stacked) MoE checkpoint would still be
refused at load. The MoE run gate needs a fitting checkpoint **whose routed
experts are per-expert NVFP4**, or the owed stacked/bf16 arm implemented first.
For the DENSE arm a fitting bf16 checkpoint is sufficient.

## Risks

- **Regression on gated rows.** These loaders serve 27B/35B/Coder. A prefix bug
  breaks checkpoints we currently gate. Mitigated by byte-identical golden md5,
  not by a green suite.
- **Half-resolved namespace.** Probing per lookup instead of once could load a
  mixture. Mitigated by design point 1 and a test with a deliberately mixed
  index, which must be refused. Refusing where upstream's `WeightsMapper`
  NORMALIZES is a deliberate divergence in the strict direction and is recorded
  as such in [porting-inventory](../porting-inventory.md) §9 deviation 17(c).
- **Untestable scale.** 92 layers / 512 experts is far past anything we can
  instantiate. Mitigated by testing config resolution and name mapping directly,
  and by *not* claiming the checkpoint runs.
- **Ahead-of-pin drift.** The forward-ported arm could diverge if upstream
  changes it before our next sync. Recorded in
  [porting-inventory](../porting-inventory.md) §9 deviation 17 as ahead-of-pin,
  with the two arms added to its §5 Qwen3.5 row, so the next sync cycle
  reconciles it deliberately.

## Tests

1. Architecture dispatch: a flat Qwen3.8-shaped config resolves
   `Qwen3_5MoeForCausalLM` to the MoE registration, and `Qwen3_5ForCausalLM` to
   the dense one. RED first — today both raise unsupported.
2. Config resolution on the real 3.8 shape: flat doc, no `vision_config`, no
   `mrope_section`; assert the scale fields and the 0.25 rotary default.
3. Weight-name mapping: a clean (`model.`) index and a VL-prefixed
   (`model.language_model.`) index both resolve every expected backbone tensor
   name; a mixed index is refused.
4. Loader byte-equality, DENSE and MoE. Two synthetic one-layer checkpoints
   with byte-identical payloads and only the namespace differing must load to
   byte-identical weights through the production `LoadQwen3_5Dense` and
   `LoadQwen3_5Moe`. The MoE case runs on BOTH expert-residency paths —
   `shards_owner == nullptr` (eager) and non-null (deferred), with
   `load_layer_experts` actually driven — because the deferred closure captures
   the resolved prefix by value and executes after the resolving frame returns,
   which is a third prefix site the dense loader has no analogue of.
4c. Refusal (added 2026-08-12): a synthetic checkpoint in the PUBLISHED shape —
   3-D stacked `mlp.experts.gate_up_proj` / `.down_proj`, bf16, no scale tensors
   — must be refused by `LoadQwen3_5Moe` with a message that NAMES the offending
   tensor and the required per-expert NVFP4 layout, in both namespaces; likewise
   a per-expert-but-unquantized index and an NVFP4 index with a bf16 `lm_head`.
   The supported per-expert NVFP4 layout must still load unchanged, asserted in
   the same case so the gate cannot be satisfied by refusing everything.
5. Inertness: 27B/35B/Coder suites unchanged, golden md5 unchanged. The
   per-layer seam DEFAULT is pinned by DRIVING `LoadQwen3_5MoeLayer` /
   `LoadQwen3_5DenseLayer` with the prefix argument OMITTED — asserting the two
   named constants does not pin it, and flipping both defaults VL->flat left the
   original case green (review finding F7).
6. Record count (added 2026-08-13): the `MODEL` ratchet bump 362 -> 364 in
   `scripts/check-agent-record.py` is tied to the two rows behind it —
   `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm` and
   `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` each appear exactly once in
   `.agents/model-matrix.md`, and the pin equals the MODEL rows that file
   carries. The existing ratchet test moves the pin by one, which holds for ANY
   pin value and so cannot say whether THIS value is right; these do.

## Gates

- Focused: the targets above plus the Qwen3.5 dense/MoE suites.
- Full gate on the row before push.
- **The run gate is OWED and must be recorded as owed.** 2.4T bf16 is ~4.8 TB;
  the only other released variant is `Qwen/Qwen3.8-2.4T-A95B-FP8` at ~2.4 TB;
  GB10 has 128 GB unified and no smaller Qwen3.8 sibling exists. There is
  therefore **no token-exact oracle run for this checkpoint**, and the row may
  not reach `DONE` on the strength of dispatch and mapping tests. What this row
  can honestly claim is that the architecture is registered and the weight
  namespace resolves — nothing about generated tokens.

If a `Qwen3_5ForCausalLM` checkpoint small enough to execute appears, that
becomes the DENSE run gate and closes that axis. **It does not close the MoE
axis**: a fitting *published* (bf16/stacked) MoE checkpoint would still be
refused at load, so the MoE gate needs one whose routed experts are per-expert
NVFP4, or the owed stacked/bf16 arm implemented first (see
[What this row does NOT make loadable](#what-this-row-does-not-make-loadable)).

## Evidence required

- RED capture of the dispatch test before registration.
- A mutation capture per prefix site — dense and MoE, including the deferred
  expert closure — showing the hardcoded VL literal makes the flat load throw.
- Green focused + full gate after.
- RED capture of the refusal case before `CheckMoeExpertLayoutSupported` exists
  (the literal `qwen3_5 weights: expected U8 for lm_head.weight`), plus a
  mutation per refusal branch and one that refuses unconditionally, which must
  turn the SUPPORTED-layout assertions red.
- Golden md5 before/after for 27B/35B/Coder showing no drift.
- The owed run gate recorded explicitly in the row and in `docs/STATUS.md`.
- Executable mutation evidence for the checker change, in
  `tests/scripts/test_agent_record.py`, which is what `scripts/check-pr-size.py`
  runs BASE-checker-against-HEAD-tree: the literal RED
  `AssertionError: 364 != 362 : the MODEL pin must equal the MODEL rows
  model-matrix.md carries` with the pin at its BASE value, a second RED from
  deleting one of the two new matrix rows, and green after both restorations.

## Stop conditions

- If the prefix cannot be resolved once per checkpoint without touching the
  per-tensor lookup contract, stop and return `NEEDS_DECISION` rather than
  scattering fallbacks through the loader.
- If any 27B/35B/Coder golden md5 moves, stop — that is a regression on a gated
  row, and this row carries no evidence that could justify it.
- Do not implement an MTP arm, a stacked/bf16 MoE expert arm, or a GGUF arm for
  3.8 on speculation; refuse them with a message naming the missing piece and
  record them as owed. (The QUANTIZED arm is the one that IS implemented — the
  earlier wording here had this inverted.)

## Now

Both rows are `PARTIAL` (2026-08-12). Registration, the once-per-checkpoint
backbone-namespace resolution and the tests above are landed on
`row/MODEL-QWEN38-TEXT-ONLY`; full CPU gate green (396/396, 1 skipped:
`test_voxtral_e2e`, fixture absent) and `tests/parity/goldens` md5-unchanged.

**Next step is the OWED run gate, and nothing else advances these rows.** The
DENSE one needs a `Qwen3_5ForCausalLM` checkpoint that fits GB10; the MoE one
needs a fitting checkpoint whose routed experts are PER-EXPERT NVFP4, because a
published (stacked/bf16) MoE checkpoint is refused at load. Neither exists
today. Until they do, the honest claim stays "the architecture is registered,
the weight namespace resolves, and an unimplemented expert layout is refused by
name".

Also owed, and deliberately NOT implemented on speculation: the **bf16 /
3-D-stacked MoE routed-expert arm** (this was recorded INVERTED as "the
quantized arm is owed" until 2026-08-12 — the quantized arm is the only one
implemented), and the MTP and GGUF arms for 3.8.

## Outcome

**Measured.** Architecture dispatch for both strings, config resolution on the
PUBLISHED `Qwen/Qwen3.8-2.4T-A95B` `config.json` (committed verbatim as
`tests/vllm/models/fixtures/qwen3_8_2_4t_a95b/config.json`, md5
`303dc59227f1d03afc941646e8df3132`) — the scale fields, the 92-entry
`layer_types` list and its `[linear,linear,linear,full] x 23` pattern, the NESTED
`rope_parameters` block both loaders' rope actually reads, and the absence of
`text_config` / `vision_config` / `mrope_section` — and weight-namespace
resolution on a clean index, a VL-prefixed index, a vision-inclusive VL index, an
index carrying `mtp.*`, a mixed index and an empty one. The strongest of these is
not a name-mapping assertion: two synthetic one-layer checkpoints with
byte-identical payloads and only the namespace differing load to byte-identical
weights through the production `LoadQwen3_5Dense` — and, on the MoE arm this row
exists for, through the production `LoadQwen3_5Moe` on BOTH expert-residency
paths, the deferred `load_layer_experts` closure included. Each of the three MoE
prefix sites was reverted to the hardcoded VL literal in turn and each RED is the
flat checkpoint failing to bind: `layers.0.input_layernorm.weight` (per-layer
base), `embed_tokens.weight` (top level) and `layers.0.mlp.experts.0.gate_proj
.weight` (the deferred closure).

**Rejected.** A per-lookup namespace fallback — it would let a checkpoint bind
half its tensors from each namespace and still appear to load, which is exactly
the failure a name-mapping test cannot see. Also rejected: a blanket
"starts with `model.`" probe, because `model.visual.*` on a vision-inclusive 27B
checkpoint would have made it look like a flat text checkpoint and turned a
checkpoint we gate today into a refusal. Only the three structural backbone
spellings vote.

**Why the defaults are what they are.** The per-layer public seams
(`LoadQwen3_5MoeLayer`, `LoadQwen3_5DenseLayer`) default `backbone_prefix` to the
VL spelling, so every 27B/35B/Coder caller is byte-identical by construction
rather than by re-measurement. The text-only arms register with
`kQwen3_5TextInfo` (hybrid YES, multimodal NO) because upstream's
`Qwen3_5ForCausalLMBase` inherits `IsHybrid` but not `SupportsMultiModal`; the
`ForConditionalGeneration` wrappers remain the multimodal registrations.

**What was NOT established, and what an earlier revision wrongly claimed.** Any
claim about generated tokens, memory or speed for `Qwen/Qwen3.8-2.4T-A95B` —
that checkpoint cannot be executed on this hardware and was never run. And,
corrected 2026-08-12 after a review FAIL, **any claim that hardware size is the
only thing between this code and that checkpoint**: it is not, because the MoE
loader reads only per-expert NVFP4 experts and both published Qwen MoE repos
ship 3-D stacked, unquantized ones. That arm is OWED and is now refused by a
message naming it (`CheckMoeExpertLayoutSupported`), with the fixture and the
literal RED in `tests/vllm/models/test_qwen3_8_text_only.cpp`. The DENSE loader
routes BF16/FP8/NVFP4 by tensor presence and is not subject to that gap — the
asymmetry is deliberate record, not an oversight.

**The refusal was verified against the REAL gated checkpoint, and its `mtp.`
exclusion is LOAD-BEARING.** Added 2026-08-12 after an independent review. The
published `nvidia/Qwen3.6-35B-A3B-NVFP4` safetensors index was fetched and read
directly (`model.safetensors.index.json`, **124,468 tensors**): it **does**
contain the exact 3-D stacked spelling `CheckMoeExpertLayoutSupported` refuses,
as `mtp.layers.0.mlp.experts.gate_up_proj` and `mtp.layers.0.mlp.experts
.down_proj` — and only there. Under the resolved backbone
(`model.language_model.`) there are **zero** stacked expert names, **zero**
expert `.weight` without a `_scale` sibling, and both `lm_head.weight_scale` and
`lm_head.weight_scale_2` are present. So the checkpoint we gate today is **not**
refused — but only because the scan is anchored at `<backbone>layers.`
(`qwen3_5_weights.cpp:633,638`) and `mtp.` is under neither backbone spelling.

That exclusion was pinned by nothing in-tree. Broadening the scan to every
`.mlp.experts.` name would refuse the one checkpoint this arm is gated on, on a
**CUDA-only load path**, with the entire CPU suite still green. The supported
fixture (`MoeOneLayerSpecs`) therefore now carries those two `mtp.` names, so
the shape of the real index is what the inertness assertions run against, and
case 4c gains a subcase that both re-asserts the fixture still carries them
(count `== 1`, and zero under either backbone prefix) and that the load stays
clean. RED-first: with the `<backbone>layers.` filter dropped, the new subcase
fails with `3-D stacked routed experts are not implemented ... found
"mtp.layers.0.mlp.experts.gate_up_proj"` (2 cases failed, `Status: FAILURE!`,
assertion count 747 → 277 as the thrown cases abort); `src/` restored
byte-for-byte afterwards and back to 7/7, 747/747.

**What the "refused by name" guarantee does and does not cover.** It covers the
routed experts and `lm_head` only: the stacked spelling, a per-expert `.weight`
with no `.weight_scale` beside it, and an unquantized `lm_head` that is present
(a checkpoint with no `lm_head.weight` at all is the tied-head case and is
deliberately not this refusal). Everything else on the MoE path still surfaces
its raw loader error — a bf16 **shared** expert, a bf16 (FP8-less) attention
tower, the compressed-tensors `weight_packed` spelling, and a tied-head MoE
checkpoint. No surface claims otherwise, so this is a clarification of scope
rather than a gap; widening the refusal belongs with the owed stacked/bf16 arm,
which has to read those layouts anyway.

**Recorded as tracked debt** in [porting-inventory](../porting-inventory.md) §9
deviation 17, with the two arms carried on its §5 Qwen3.5 row and the owed run
gate on [BENCHMARKS](../../docs/BENCHMARKS.md) §Open gaps: the ahead-of-pin
anchor `ad5d29db7` (17a/b), the deliberate REFUSAL of a mixed namespace where
upstream's `WeightsMapper` would normalize it (17c), the published config's
transformers-4.57.3 `dtype` key, which `hf_config.cpp:520-522` does not consume
(17d — inert, no reader, and a fix would touch every model, so it is pinned by
an assertion rather than smuggled in here), and the **unimplemented bf16 /
3-D-stacked MoE routed-expert arm** (17e, added 2026-08-12 — it was previously
recorded inverted, as the quantized arm being the owed one).

**The safetensors alignment class** ([#627](https://github.com/mudler/vllm.cpp/issues/627),
fixed 2026-08-13). This row's `test_qwen3_8_text_only` was the first gate ever to
run a safetensors weight loader under UBSan, and it went RED on
`qwen3_5_weights.cpp:298` — `load of misaligned address ... for type
'const uint16_t', which requires 2 byte alignment`. **The defect is pre-existing
(`8ee2c0766`), not this row's**; what this row supplied is the first synthetic
checkpoint whose tensor offsets are not all even, which is a legitimate shape a
real file can have because a safetensors offset is just the running byte total of
everything ahead of it.

The observed site was one instance of a class. A sweep of every
`reinterpret_cast<T*>(<sttensor>.data)` in `src/vllm/model_executor/models/`
found **fifteen** across nine `*_weights.cpp` loaders, including two with
stricter-than-2-byte requirements: `olmo2_weights.cpp` forms a `const float*`
(4-byte) and `qwen3_dspark_weights.cpp` a `const int64_t*` (8-byte). All fifteen
now go through `vt::LoadUnaligned` — the seam `ea4deb203` introduced and that
`dense_loaders::TransposeBf16` and `minimax_h3_vae_loader.cpp:87` already used;
the local `qwen3_5_weights.cpp` copy of `TransposeBf16` had simply never been
migrated to it. Two sites could not take a byte pointer and were handled in kind:
`internlm2_weights.cpp` only ever bulk-`memcpy`s from its source, so it keeps a
`const uint8_t*` and scales its offsets, and `gemma4_weights.cpp` feeds a typed
scale pointer to `DequantFp8ChannelToBf16` (whose header is outside this row's
authority), so it copies the N-element scale row into an aligned buffer first.

**Inertness is proven three ways, not asserted.** (1) `vt::LoadUnaligned<T>` is
`memcpy`, which is bit-identical to `*(const T*)p` on every input the old code
was *allowed* to read — a scratch harness ran the original and rewritten form of
all six loops over one payload at an aligned base and got byte-identical output,
then reproduced those same bytes from the rewritten form at misaligned bases 1
through 8. (2) `tests/parity/goldens` is untouched and every golden-comparing
suite passes. (3) Full CPU gate 404/404 (2 skipped: `test_voxtral_e2e` and
`test_modelopt_mixed_precision_checkpoint`, fixtures absent) and — the gate that
was red — full ASan+UBSan 404/404, `test_qwen3_8_text_only` back to 7/7 and
747/747 with zero runtime errors.

**No load-time regression, checked rather than assumed.** `TransposeBf16` is a
hot load-time loop and a naive per-element `memcpy` is exactly the kind of change
that can turn one load into a call. It does not here: at `-O2` the old and new
inner loops are instruction-for-instruction identical — the same six
instructions, the same `movzx REG, WORD PTR [rax]`, no call emitted. The whole
delta is five prologue instructions (one callee-saved push/pop pair and two
address setups), paid once per call, not per element.

**Still owed, and deliberately not touched here** because they fall outside this
row's authority (`*_weights.cpp`): the same cast survives at
`voxtral.cpp:51,347`, `qwen3_vl.cpp:78` and `qwen3_5_mtp.cpp:71`. The first three
are genuine misaligned *loads* of the same severity as the one UBSan caught; the
`qwen3_5_mtp.cpp` one only forms the pointer and then `memcpy`s through it, so it
is UB but will not fire the `alignment` check. None is reached by any suite that
runs under sanitizers today, which is precisely why they need a follow-up rather
than a grep.
