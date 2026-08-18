# LTX-2.5 — `keyframes_abs_pos_embedding`, the module that refuses both shipped checkpoints

Row: `LTX25-KEYFRAMES-ABS-POS`
Issue: [#658](https://github.com/mudler/vllm.cpp/issues/658)
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644)
Pin: `Lightricks/LTX-2 @ fd4ded7f`

## 0. What is wrong today — measured, not inferred

**Neither shipped LTX-2.5 DiT can be loaded inside the contract.** Both are
refused, from opposite directions, and both refusals are correct given that the
module is unported. Two renders on `dgx.casa` established this, and the
checkpoint headers were read directly to confirm why:

| checkpoint | header facts | refusal |
|---|---|---|
| `vonkaiser-fp8-nvfp4/.../ltx-2.5-22b-distilled-fp8.safetensors` | 6124 tensors, **no `__metadata__` at all**, carries `model.diffusion_model.keyframes_abs_pos_embedding` `F8_E4M3 [1,4096]` + `F32` scale | "carries modules this port does NOT carry: `keyframes_abs_pos_embedding`" |
| `lightricks-ltx-2.5/.../ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 7876 tensors, `__metadata__` present, **declares `use_keyframes_abs_pos_embedding=true`**, carries **no** keyframe tensor | `ltx2.cpp:192` — "`use_keyframes_abs_pos_embedding` is not ported … the checkpoint does not carry `keyframes_abs_pos_embedding`" |

So the only way to render from a real checkpoint today is
`allow_unported_modules`, which loads the ported subset. That is honest about
what it drops, but it means **every render we can currently produce is missing a
trained term** — see §1. "It renders with the escape hatch" is not the port
being complete.

**This is a mirror, not a product decision.** The reference defines the
behaviour exactly, so it gets ported; nothing here is escalated.

## 1. What upstream does, with anchors

**The parameter** — `model/transformer/model.py:215-219`:

```python
# Marks tokens whose latent encodes a single standalone pixel frame. Zero-initialized, so a
# checkpoint that predates it behaves identically until the parameter is trained.
self.keyframes_abs_pos_embedding = (
    torch.nn.Parameter(torch.zeros(1, self.inner_dim)) if self.use_keyframes_abs_pos_embedding else None
)
```

**The consumer** — `model/transformer/transformer_args.py:38-43`, called once at
`:269`:

```python
embedding = embedding_provider()
if embedding is None:
    return hidden_states
mask = (keyframes_mask > 0).to(dtype=hidden_states.dtype)
return hidden_states + mask * embedding.to(dtype=hidden_states.dtype)
```

**The mask** — `tools.py:184` plus `_first_frame_keyframes_mask` (`:186-195`)
marks the target's **first latent frame unconditionally**. Upstream's own comment
says so in terms: *"the reference implementation marks it unconditionally --
independently of whether any keyframe slots exist."*

**Therefore, on a trained checkpoint, upstream adds a learned `[1, inner_dim]`
per-token bias to every token of the first latent frame, on every forward,
whether or not a keyframe was supplied.** The module is ~three lines of maths; it
is small and it is unconditional, which is exactly why its absence is easy to
miss and expensive to leave.

**Do not port `enable_keyframes_abs_pos_embedding` / `supports_…`
(`model.py:166-173` and `:175-200`).** Both are **defined and never called**
anywhere in the checkout — one grep hit each, the definition, re-confirmed with a
positive control in the same command. Re-verify before relying on it; do not take
it from this spec. But **read `supports_…`**: §2 shows it returns `False` on the
NVFP4 arm both before and after the load, which is what settles that arm's
behaviour even though nothing calls it.

## 2. Why this is not "just a zero"

The zero-init comment above is what made this look inert, and it is the trap.
Two independent reasons it is not:

1. **On the FP8 DiT the tensor is trained and present.** `F8_E4M3 [1,4096]` with
   an `F32` scale is not `torch.zeros`.
2. **On the NVFP4 DiT the parameter is never materialised at all.** Upstream
   builds on the **meta device** (`loader/helpers.py:90-91`, `create_meta_model`)
   and loads with `assign=True` (`loader/single_gpu_model_builder.py:98`), so an
   absent key stays on `meta` — reading it raises. A reviewer executed this. It
   is not a zero that is harmlessly added; it is a parameter that does not exist.

The polarity does mean a *genuine* zero would be inert, since the term is
**added**, not multiplied. That is why the FP8 case is the one that changes
output.

**SETTLED BY EXECUTION on 2026-08-13, and the answer decides the NVFP4 arm.**
Two reviewers had disagreed — one read the parameter as *zero-initialized*
(`model.py:200`) and therefore "an exact no-op there"; the other **ran** it. Both
readings are consistent with the source; only one is consistent with what runs.
An implementer then executed upstream's own `create_meta_model`
(`loader/helpers.py:84-95`) against the real NVFP4 `__metadata__`, read through
upstream's own `read_model_metadata` / `SafetensorsModelStateDictLoader`
(`sft_loader.py:58-74`):

```
keys matching 'keyframes_abs_pos' in the file : []   (0 of 7876 entries)
config.transformer['use_keyframes_abs_pos_embedding'] = True
keyframes_abs_pos_embedding: shape=(1, 4096) f32 device=meta is_meta=True
supports_keyframes_abs_pos_embedding (BEFORE load) : False
after load_state_dict(sd, strict=False, assign=True):
  neighbour patchify_proj.weight : device=cpu is_meta=False   <- materialised
  keyframes_abs_pos_embedding    : device=meta is_meta=True
  in missing_keys                : True
  reading the value RAISES : RuntimeError: Tensor.item() cannot be called on meta tensors
supports_keyframes_abs_pos_embedding (AFTER load)  : False
```

The materialised **neighbour** is what makes this mean something: the loader did
run and did populate the model; only the absent key stayed on `meta`.

**Therefore, on the first-party NVFP4 DiT, upstream never reaches the add at
all** — `supports_keyframes_abs_pos_embedding` is False before *and* after the
load. So the correct mirror on that arm is **to load and apply nothing**, which
is neither a refusal nor an invented zero. Refusing it, as `ltx2.cpp:192` does
today, is stricter than upstream; synthesising a zero and adding it would be
inventing behaviour that upstream's own guard exists to prevent.

Two consequences for this row. **The NVFP4 arm's refusal is retired outright**,
not replaced by a no-op path with a fabricated tensor. And a render taken today
on the NVFP4 DiT with `allow_unported_modules` is, **for this module only**,
upstream-equivalent — the flag is declared, the tensor is absent, and upstream
would apply nothing either. That is a narrow claim about one module and must not
be restated as "the render is upstream-equivalent".

**The flag is not where you would look for it.** A raw read of `__metadata__`
returns `None`; it lives at `config.transformer`, which is why upstream's
JSON-decoding reader is needed to see it at all.

**Re-derive these anchors at HEAD before relying on them.** The guards are
`model.py:166-173` and `:175-200`, with the quoted phrases at `:170` and `:182`
— corrected from an earlier brief that said `167-182` / `175-193`.

**What is not in dispute, because it was measured on the bytes:** the vonkaiser
FP8 copy is **trained** — `F8_E4M3 [1,4096]`, **4096 of 4096 bytes non-zero**,
plus its `F32` scale. Whatever the NVFP4 arm turns out to do, that arm changes
output.

## 3. Scope

**In.**

1. Port the parameter and its application: load `keyframes_abs_pos_embedding`
   (and its scale on the quantized arm), and add `mask * embedding` at the same
   point upstream does.
2. Port the mask rule: the target's **first latent frame** is marked
   unconditionally, mirroring `_first_frame_keyframes_mask`.
3. Retire the two refusals — `ltx2.cpp:191-196` (flag declared, tensor absent)
   and the loader's "carries modules this port does NOT carry" entry — and
   remove the force-clear of the flag under `allow_unported_modules`
   (`ltx2_loader.cpp:974-981` / `:1019-1022`; re-derive these anchors at HEAD,
   they have moved twice this campaign).
4. Correct `include/vllm/model_executor/models/ltx2.h:47-49`, which says *"LTX-2.5's
   checkpoint does not carry the parameter"* — false for the FP8 DiT.

**Out.**

- Keyframe *conditioning* as a user-facing feature (supplying keyframe slots).
  This row is the unconditional first-frame bias only, which is what both
  checkpoints need in order to load at all.
- Any change to the image-conditioning arm's own behaviour beyond gaining the
  bias it should always have had. That arm is [#657](https://github.com/mudler/vllm.cpp/pull/657)/[#666](https://github.com/mudler/vllm.cpp/pull/666).

## 4. Memory format — check this explicitly

Upstream casts **both** operands to `hidden_states.dtype`
(`transformer_args.py:42-43`). There is no `f32` escape on this path. Mirror that
polarity: the bias is applied in the model dtype. **A token gate cannot catch a
dtype that is too wide** — it stays numerically correct while moving twice the
bytes — so check the memory format against the reference explicitly rather than
inferring it from a passing golden.

On the FP8 arm the tensor is `F8_E4M3` with an `F32` scale: dequantize per the
existing tower convention, and state which one you used.

## 5. Tests

RED-first, and the RED must be *the intended failure*, not a load refusal.

1. **A golden from upstream**, generated by executing `ltx_core` at the pin on a
   fixture with a known `keyframes_mask`, asserting the first latent frame's
   tokens differ from the rest by exactly the bias and that every other token is
   untouched.
2. **The unconditional rule**: a generation with **no keyframe supplied** must
   still mark the first latent frame. This is the half most likely to be ported
   as "only when keyframes exist", which would be silently wrong on every render.
3. **Both real checkpoints load without `allow_unported_modules`.** Env-gated on
   `LTX2_CHECKPOINT_ROOT`, and note in the report that CI does not set it — CI
   runs ~5.7% of `test_ltx2_video`'s assertions, so this test is host-local
   evidence, not a gate.
4. The two retired refusals must no longer fire, and the tests that asserted them
   must be **replaced by tests of the new behaviour**, never merely deleted.
   Deleting an assertion to turn a gate green is forbidden.

**Mutations that must be run and recorded:** zero the bias (golden must go RED);
apply the mask to the wrong frame (RED); make the mask conditional on a supplied
keyframe (RED — this is test 2's whole point); apply in `f32` and store back wide
(must be caught by the memory-format check, and if it is not, say so — that is a
finding about the gate, not a pass).

## 6. Risks

- **Every existing LTX-2.5 golden changes**, because the bias is unconditional
  and non-zero on the FP8 arm. That is the correct outcome, but it means goldens
  must be **regenerated from the generator against the pin**, with the
  regeneration itself proved reproducible, and the diff explained per value
  rather than accepted wholesale.
- **The FP8 arm is the only one whose output changes.** The NVFP4 arm is settled
  in §2 by execution: upstream never reaches the add there, so this row **retires
  that refusal and applies nothing**. The risk to guard is the tempting
  middle option — allocating a zero tensor because the code path wants one, and
  adding it. That is not a mirror; upstream's own `supports_…` guard exists
  precisely to stop a model reaching the add without a trained parameter.

## 7. Stop conditions

- If porting the mask changes token counts or phase geometry, stop: that is a
  different row.
- If the FP8 dequant convention for a `[1,4096]` + scalar-scale tensor does not
  already exist, stop and report rather than inventing a second convention.
- `dgx.casa` is required only for the two real-checkpoint loads, and it rebooted
  three times on 2026-08-13; treat GPU evidence as best-effort and gate the row
  on the fixture goldens.

## Outcome

Implemented 2026-08-14 on `row/LTX25-KEYFRAMES-ABS-POS`.

**§2 reproduced, and it is now a script rather than a paragraph.**
`scripts/measure-ltx2-keyframes-meta.py` runs upstream's own `create_meta_model`
+ `SafetensorsModelStateDictLoader` against the real NVFP4 file at pin
`fd4ded7f`:

```
declared - state_dict          : 1 key(s)
    keyframes_abs_pos_embedding
loadable at declared shape     : 2914 of 7876
neighbour scale_shift_table    : device=cpu is_meta=False   <- the CONTROL
keyframes_abs_pos_embedding    : device=meta is_meta=True
in missing_keys                : True
reading the value RAISES       : Tensor.item() cannot be called on meta tensors
supports_... BEFORE / AFTER    : False / False
```

Two corrections to §2's recipe, both discovered by running it. The control key
had to change: `patchify_proj.weight` is NVFP4-**packed**, half its logical
width, so `load_state_dict` raises on it and the state dict must be filtered to
the unpacked parameters first — `scale_shift_table` is the neighbour that
actually materialises. And the load needs upstream's own
`LTXV_MODEL_COMFY_RENAMING_MAP` (`model_configurator.py:222-226`); without it
every key keeps its `model.diffusion_model.` prefix, nothing matches, and the
probe reports "still on meta" for the WHOLE model — an instrument that agrees
with the conclusion for the wrong reason.

**The FP8 header, re-measured on the bytes:** `F8_E4M3 [1, 4096]`, 4096 of 4096
bytes non-zero, plus `keyframes_abs_pos_embedding_scale` `F32` rank-0. That
scalar-scale shape is exactly what `MaterializeDitTensor`'s existing `F8_E4M3`
arm reads (`ReadScalarF32` + `DequantFp8ToBf16` → `kBF16`), so the stop condition
in §7 did not fire: **no second dequant convention was invented.**

**Resolution rule.** `Ltx2DitParams::use_keyframes_abs_pos_embedding` means
`supports_keyframes_abs_pos_embedding` (`model.py:166-173`), not the raw config
flag. `Ltx2AdoptDeclaredDitParams` resolves a declared flag against what the file
carries; the `allow_unported_modules` force-clear is gone, and so is that
parameter — a render's opt-in must not decide a correctness question. Three
outcomes, all upstream's: declared+present → applied; declared+absent → nothing
applied, no refusal; not declared → nothing applied. The **rejected** fourth is
§6's tempting middle: allocating a zero tensor because the code path wants one.

**Goldens.** `gen-ltx2-goldens.py` gains section 7, which TRAINS the parameter
rather than leaving it at upstream's zero-init — a zero bias is an exact no-op
because the term is ADDED, so a zero arm would gate nothing. Regeneration is
reproducible: two runs at the pin differ only in the recorded argv line, and the
run against the pre-change generator reproduced the checked-in file
byte-for-byte on the same one line. §6's risk did **not** materialise as feared:
the diff is +650/−8 and **all eight removed lines are `// --- forward case ...`
banner comments** that gained two fields. **Zero existing golden VALUES changed**,
because the deterministic weight stream is keyed by parameter NAME, so adding a
parameter perturbs nothing else. Measured magnitude: the marker moves the DiT
video output by 0.278 (71.5% of max|unmarked|) and the audio output by 1.49%,
the latter only through the audio↔video cross attention.

**Three gate holes, found by mutation and closed in-flow.** The third one
falsified the first, so hole 1 is recorded here with the correction it needed
rather than with the claim it originally made.

1. **A CONDITIONAL marker was invisible — at the CONSTRUCTION site.** Wrapping
   the engine's mask in `if (wants_image)` compiled clean and left all five
   LTX-2.5 suites GREEN (43/43, 17/17, 28/28, 35/35, 36/36). Two causes: the
   fixture DiT did not carry the parameter, so no engine render could observe a
   drop; and an empty `std::vector`'s `data()` is `nullptr`, which is upstream's
   *legal* "no token is marked" — the defect arrives wearing a supported path's
   costume. Fixed by giving `ReducedDitParams` the flag (as the shipped FP8 DiT
   resolves it) and by refusing an empty marker by name. Re-run: 11 RED cases in
   `test_ltx2_video`. **That fix closed one of two sites, and its headline claim
   was therefore only half true** — hole 3 is the other half, and it was found by
   the first fresh review rather than by this row.
2. **Nothing looked at the DEVICE addend's WIDTH.** Removing the
   `dtype == out.x->t().dtype` equality left all five suites green. That is the
   too-WIDE case exactly: an f32 addend under a bf16 stream is numerically
   correct, agrees with every golden, and moves twice the bytes. A dedicated case
   now stages at bf16, swaps in the f32 view of the same values, and requires a
   refusal by name — with a positive control for its own substring search.
3. **The marker was gated where it is BUILT, not where it is HANDED OVER.** The
   HANDOVER is a second, independent way for the marker not to reach the forward,
   and hole 1's guard did not cover it. That guard reads
   `video.keyframes_mask` — the VECTOR — at `ltx2_video.cpp:1682`, so it fires
   when the mask is built conditionally and stays silent when the ASSIGNMENT one
   line lower is: writing `if (wants_image) vin.keyframes_mask =
   video.keyframes_mask.data();` at `:1688` compiled clean and left all five
   suites byte-identically GREEN while the rendered pixels moved — frame 0 went
   from a flat 127 to a flat 130. Two sites, one invariant.

   Closed at two altitudes. At the SEAM, a second `VT_CHECK` at `:1707` reads
   `vin.keyframes_mask`, the field `Ltx2DitForward` actually consumes, AFTER the
   handover rather than the vector before it. In the TEST, `test_ltx2_video`
   gained the absolute output assertion it had no version of: every check in that
   suite was relative (`digest != digest`, `absmax > 0`, "not one flat value"),
   which is exactly why a term that silently vanished from every render was
   invisible to it. The new case renders twice from two checkpoints that differ in
   exactly one thing — `EnumerateLtx2DitTensors` gates exactly one entry on the
   flag (`ltx2.cpp:274-276`) and `Param()` seeds every tensor from `Fnv1a(name)`
   alone — on a request with NO image and NO keyframe, which is upstream's
   unconditional case (`tools.py:186-196`).

   MEASURED, unmutated: the marker moves **25848 of 91169** artifact bytes
   (28.4%). With the seam assert neutralised and the handover made conditional,
   the two renders collapse to **0 of 91169**, and the new case is the ONLY thing
   in the tree that reds. Compile status printed beside every mutation result — 0
   errors, links clean — so no red here is a build failure wearing a test verdict.

   Two further mutations, added by the fresh reviewer, decide the shape of the
   fix rather than confirm it. **M3** zeroes the mask CONTENT, leaving both
   `VT_CHECK`s passing, and still reds only that one case: the test closes the
   CLASS — "the trained term reached the forward" — not the two known lines.
   **M4** replaces the flag-gated seam condition with a bare `!= nullptr` and
   makes the new case THROW on the flag-off render. That is why the condition is
   `use_keyframes_abs_pos_embedding` and not a null check: upstream's
   `apply_keyframes_absolute_embedding` exits on `keyframes_mask is None`, so a
   DiT that does not carry the parameter MUST reach the forward with a null
   marker, and handing that model a marker is a refusal at `ltx2_dit.cpp:520`,
   not a fix.

**Not in scope and still owed:** keyframe *conditioning* as a user feature
(supplying slots), which needs the token-APPEND machinery `ltx2_video.cpp`'s
last-frame refusal describes. The DiT FORWARD on the shipped 21.00B geometry is
still out of reach on this box.

**No real-checkpoint evidence for this row is GATED, and none was produced by a
gate run.** Say it plainly rather than let "stops at load" imply a load was
gated. The three env-gated suites (`LTX2_CHECKPOINT_ROOT`, `LTX2_FP8_DIT`,
`LTX2_SHIPPED_DIT`) appear in no workflow (#673) and printed `SKIPPED` in every
implementation and review run, so nothing in CI has ever opened either shipped
checkpoint for this row — not even to load it. The real-checkpoint facts recorded
above come from two host-local instruments run by hand: the safetensors header
reads, and `scripts/measure-ltx2-keyframes-meta.py` against the NVFP4 file at pin
`fd4ded7f`. Both are reproducible and neither is a gate.

## Now

`DONE`, reviewed `PASS`. The fresh review reproduced the decisive mutation table
at `46a096715`, added M3 and M4 above, and confirmed the seam check is correctly
conditioned, the marked/unmarked pair is a one-variable experiment, and no third
silent path to the forward exists.

The reviewed history could not land. Two of its commits carry
`Assisted-by: AGENT:claude-opus-5[1m] [claude-code]`, which
`scripts/check-commit-trailers.py` rejects because `ASSISTED_BY` requires a space
before a bracket group, and two more touch `src/vllm/model_executor/models/`
without editing `docs/FEATURES.md`, which `scripts/check-doc-checkpoint.py`
rejects. Both checkers walk the branch's own commits, both are required checks,
and repairing a commit message rewrites history, which `AGENTS.md` forbids
pushing over a branch. So the content was rebuilt on
`row/LTX25-KEYFRAMES-ABS-POS-V2`. Applying `46a096715`'s tree onto `main` gave
`git write-tree` = `8c188b13eb3d5789eb0bd2dddd3da423f83a7101`, byte-identical to
`46a096715^{tree}`, and the replacement then differs from that head in exactly
three files: this `## Outcome` and `## Now`, the issue-index row it repairs, and
one word in `docs/FEATURES.md`. That word is `SERVED`, which fell out of "IMAGE
cond SERVED `crf=0`" when this row's rewrite hit the 220-character
`MAX_CELL_CHARS` cap and then appeared nowhere in `FEATURES.md` or `STATUS.md`. A
budget is the wrong author for deciding which fact survives, so it is restored at
the cost of two redundant words in the same cell, and the entry is 217
characters. `row/LTX25-KEYFRAMES-ABS-POS` was closed as a record. The operator
reruns the gate on the replacement and lands it.
