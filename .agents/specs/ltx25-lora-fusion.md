# Fuse N LoRA adapters, because upstream's aggregator takes a list and its second product rounds differently from its first

Row: `LTX25-LORA-FUSION`.
Issue: [#932](https://github.com/mudler/vllm.cpp/issues/932) — the `## Owed`
entry "N-adapter fusion, which additionally needs upstream's SECOND rounding
pattern (`addmm_` with `alpha`, `fuse_loras.py:115`) that this row refuses
rather than guesses" in
[`ltx25-ic-lora.md`](ltx25-ic-lora.md). Order 8, item **A17** of
[`ltx25-completion-scope.md`](ltx25-completion-scope.md).
Predecessor: [`ltx25-ic-lora.md`](ltx25-ic-lora.md) (the one-adapter arm) and
[`ltx25-lora-fuse-seam.md`](ltx25-lora-fuse-seam.md) (the `vt::Matmul` routing).

**`gh` cannot read #932.** `gh api repos/mudler/vllm.cpp/issues/932` returns
`404` in this session, as do #930 and #975, while #2585 resolves — the scattered
blindness #435/#644/#1854/#2295 record. `REMOTE_UNVERIFIED` is not absence, and
#930 is cited by the predecessor spec as landed in `c7cb59fbb`, so the low range
exists. The issue is referenced and not re-filed.

**#932 OWNS TWO ARMS, THIS ROW LANDS ONE, AND SO NOTHING HERE CLOSES IT.** The
same issue is the recorded owner of **A16**, `conditioning_attention_strength <
1.0` and its attention mask, in three committed places a reader has: the
`## Owed` entry beginning "the `conditioning_attention_mask` /
`conditioning_attention_strength < 1.0` arm" in
[`ltx25-ic-lora.md`](ltx25-ic-lora.md), the **A16** row of
[`ltx25-completion-scope.md`](ltx25-completion-scope.md), and a SHIPPED
user-facing refusal in `ltx2_video.cpp` whose text reads "The sub-1.0 arm is
owed by #932, and it is not what blocks this one". A16 is out of scope here
(§4), is still unimplemented, and its refusal is still live, so a closing
keyword on this row would take A16's only owner away and leave a shipped error
message pointing a user at a closed issue. `scripts/check-agent-record.py`
cannot see that: it asks only that a referenced issue HAS an owning row or an
`## Owed` entry, and #932 has two.

**So this row's pull request carries no closing keyword** and names the half it
lands instead. The alternative — giving A16 its own issue and retargeting the
three references — needs `ltx25-completion-scope.md` edited, and that document
is an operator-owned record a sibling row was told not to touch, so the split is
left to the operator and #932 stays open on A16.

## 1. The refusal this row lifts, and whether its stated reason was true

`Ltx2ResolveLoraReferenceFactors` refuses a second adapter
(`src/vllm/model_executor/models/ltx2_lora.cpp`, the `adapters.size() > 1`
branch), and `Ltx2FuseLoraIntoTensor` carries a second refusal for the case that
one cannot reach. Two sibling rows in this campaign found their refusal's stated
reason FALSE, so it was checked line by line against the pin before anything was
written.

**It was true.** Every clause holds at Lightricks/LTX-2 `fd4ded7f`:

| the refusal says | read at the pin | verdict |
|---|---|---|
| "Upstream's own dubit.py enforces the same (dubit.py:364-365)" | `:364` `if not args.lora or len(args.lora) != 1:`, `:365` `raise ValueError("Dub-It requires exactly one --lora (the Dub-It IC-LoRA).")` | TRUE |
| "hdr_ic_lora.py takes exactly one (hdr_ic_lora.py:271-272)" | `:271` `lora_path = str(Path(hdr_lora).resolve())`, `:272` `loras = (LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)` — a one-tuple built from a scalar parameter | TRUE |
| "ic_lora.py does accept a list" | `ic_lora.py:75` `loras: list[LoraPathStrengthAndSDOps]` | TRUE |
| "Upstream aggregates them with a SECOND rounding pattern (addmm_ with alpha, fuse_loras.py:115)" | `:115` `aggregated.addmm_(product.b, product.a, alpha=product.strength)` against `:113` `torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)` | TRUE |

Every row of that table was read AGAIN, at the pin, by the second implementer
after a session limit ended the first one's run, because a table asserting
"TRUE" is exactly the kind of inherited claim this campaign has found wrong. It
holds. What did NOT hold on re-reading is §5's claim about the phase scope; that
section carries the correction.

So the refusal was an honest statement of a real gap, and this row closes the
gap rather than correcting a false claim. What it did NOT say, and what makes
the gap bigger than "one pipeline entry point wants a list", is in §2.

**The `--lora` CLI is the fourth reader and it has always been N.**
`utils/args.py:600-611` declares `--lora` with `action=LoraAction`, `nargs="+"`,
`default=[]` and the help text "Can be specified multiple times. Example:
`--lora path/to/lora1.safetensors 0.8 --lora path/to/lora2.safetensors`". Every
pipeline that reaches `DiffusionStage.from_checkpoint` through that flag can
therefore be handed N, and `dubit.py` and `hdr_ic_lora.py` are the two that
narrow it back to one. The arity cap mirrored the two narrowest readers and not
the flag.

## 2. The composition upstream needs and this tree could not express

`dfr_pipeline.py:212` builds `stage_loras = (*self._user_loras,
*self._distilled_lora)` and hands that tuple to the single `DiffusionStage` it
builds at `:213`, whose `loras=` argument is `:217`. That is TWO adapters on one
resident DiT whenever a user adapter is supplied alongside the required
distilled one.

This tree routes both through the SAME `lora_path` load extra
(`ltx2_video.cpp`, the `requires_distilled_lora` refusal names
`kLtx2LoraPathExtra` as where the distilled adapter goes), and that extra held
exactly one value. So the DFR and HQ arms could carry the distilled adapter OR a
detailing IC-LoRA and never both, and the arity cap was the reason. The cap is
therefore not only a missing capability on `ICLoraPipeline`; it is the reason a
shipped recipe cannot be given what upstream gives it.

## 3. The arithmetic, measured rather than transcribed

`aggregate_lora_products` (`fuse_loras.py:99-116`) has two forms:

```python
if aggregated is None:
    aggregated = torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)
else:
    aggregated.addmm_(product.b, product.a, alpha=product.strength)
```

The first is landed and gated. The second was NOT transcribed: three candidate
models of `addmm_` on a bf16 aggregator were run against the pinned module
itself, executed at `fd4ded7f` with `torch 2.11.0+cu130`. The probe was RE-RUN
independently after a session limit killed the first implementer, and this table
is the re-run's numbers, not the inherited ones:

| model | rule | matches upstream |
|---|---|---|
| **A** | `bf16( f32(agg) + strength * f32_accumulated(B @ A) )` | **YES** — 49 of 49 trials |
| B | the product ROUNDED to bf16 before the strength is applied | no — failed 47 of 49 trials; 18 of 120 elements on this row's own fixture |
| C | `strength * product` rounded to bf16 before the add | no — failed 45 of 49 trials; 21 of 120 elements on this row's own fixture |

The 49 trials are 40 randomized ones with 2 to 4 adapters and `M`, `K`, `N`
drawn from 1 to 8, plus `K` in {16, 64, 256, 1024, 4096} at `M = N = 3` and `K`
in {16, 64, 256, 1024} at `M = 6`, `N = 20`. `K` is the LoRA RANK here — `B` is
`[out, rank]` and `A` is `[rank, in]` — so 4096 is already an order of magnitude
past any published adapter, which is the reach a golden of rank 3 does not have
on its own.

**The distinguishing fact is that no bf16 rounding sits between the GEMM and the
strength.** The first form rounds `B * strength` to bf16 BEFORE its matmul; the
second applies `alpha` to an f32 accumulation and rounds ONCE, at the store. A
port that reused the first form's pre-scaling for the second would be wrong in a
way no tolerance-based gate would report, which is why B and C are recorded here
with the element counts that separate them. Substituting model B into the fuser
reds the golden at exactly 18 of 120, which is the probe's own number arriving
from the other side.

`vt::Matmul`'s contract already admits the shape this needs: "a/b float dtypes
(f32/f16/bf16), out **f32** or bf16, f32 accumulation" (`vt/ops.h`). So the
second form is the same shared seam with an f32 output tensor, and no new op is
added.

## 4. Scope

**In.**

- `Ltx2FuseLoraIntoTensor`: the second product form, and the loop over N
  adapters in the order the load supplied them (`_products_for_sd_key`
  iterates `lora_sd_and_strengths` in list order).
- `Ltx2ResolveLoraReferenceFactors`: drop the arity refusal. Its conflict
  branches (`ic_lora.py:155-173`) become reachable for the first time.
- The load extras: `lora_path_2`, `lora_strength_2`, ... `_N` beside the
  existing `lora_path` / `lora_strength`, which keep their meaning for the
  first adapter. `ltx2_gen --lora` accumulates across repetitions, mirroring
  `args.py:600-611`.
- The header and comment statements that assert the cap: `ltx2_lora.h`,
  `ltx2_loader.h`, `ltx2_pipeline.h`, `ltx2_loader.cpp`, `ltx2_pipeline.cpp`.
- The refusal §5 obliges: N > 1 adapters on a recipe with a `kNoAdapters`
  phase, refused at load by name. Lifting the cap is what makes that arm
  reachable, so refusing it is part of the same change and not a later sweep.
- `docs/FEATURES.md`, one row.
- `docs/models/ltx-2-5.md`, the page that documents `--lora`: the flag becomes
  repeatable and the indexed load extras are new config keys, which the public
  documents table owes a surface. **The surface is the model recipe and not
  `docs/USAGE.md`**, and that was checked rather than assumed: `grep -ni lora
  docs/USAGE.md` returns three lines, all in the checkpoint registry and its
  prose, and `grep -rn lora_path docs/` returned NOTHING BEFORE this change (it
  returns this row's own new lines after it) — so `--lora`, `lora_path` and
  `lora_strength` have never been in `USAGE.md`, while `ltx-2-5.md` documented
  `--lora` in four places already. `USAGE.md` routes a reader to
  the recipe under `## Find a model recipe` for exactly this: "commands,
  required weights, component-specific runtime settings, and known limits for
  each model family". A positive control, because an absence read off a failed
  grep is not a finding.

**Out.**

- **A15** (`ICLoraPipeline`, reference-image and reference-video conditioning)
  and **A16** (`conditioning_attention_strength < 1.0`), the two siblings in
  order 8. Neither is a dependency: N-adapter fusion is complete without a
  reference conditioning path, and the reference refusal
  (`ltx2_video.cpp`) names causes this row does not touch.
- Per-phase adapter STRENGTH (`ti2vid_two_stages_hq.py:92-101`), owed by #1144.
  Lifting the arity cap does not supply it, and `Ltx2PhaseLoraScope` stays a
  two-valued enum — see §5.
- Re-quantizing a fused FP8/NVFP4 weight. Unchanged from the predecessor row.

### 4.1 Why the transport is INDEXED KEYS and not a delimited list

The extras surface is `std::map<std::string, std::string>` and
`VideoExtrasFromArrays` folds the C ABI's parallel arrays into it, so a repeated
`lora_path` key silently last-wins. N adapters therefore need a spelling.

The tree's own precedent for a `nargs` flag is a comma-separated value
(`ApplyStgBlocksExtra`, mirroring `--*-stg-blocks`). **That precedent does not
transfer, because those values are integers and these are PATHS.** A comma is
legal in a POSIX filename, and the only bytes that are not are `/` and `NUL`,
neither of which can serve as a separator in a C string. A delimited encoding
would therefore split some real path in half and refuse it as two nonexistent
files, or worse, name two files that both happen to exist. Indexed keys cannot
be ambiguous, are byte-compatible with every existing caller, and make a gap
(`lora_path_3` with no `lora_path_2`) a refusal instead of a silent renumber.

This is a transport decision and not a mirrored behaviour: upstream's transport
is `argparse`, which this ABI is not.

## 5. `Ltx2PhaseLoraScope` stays two-valued, and the LOAD refuses what it cannot say

`ltx2_pipeline.h` argued that `kAllAdapters` / `kNoAdapters` is the COMPLETE
space because "`Ltx2ResolveLoraReferenceFactors` refuses more than one adapter
by name ... so the powerset of the load's adapters has exactly two members".
**That argument dies with this row and the conclusion does NOT survive it.**

The first draft of this spec claimed the conclusion survived, on the ground that
no upstream recipe scopes a proper subset of its adapters to a phase. **Read at
the pin, that is false, and the anchors it cited are the ones that falsify it:**

| recipe | stage 1 | stage 2 |
|---|---|---|
| `a2vid_two_stage.py` | `loras=tuple(loras)` (`:107`) | `(*tuple(loras), *tuple(distilled_lora))` (`:114`, taken `:116`) |
| `ti2vid_two_stages.py` | `loras=tuple(loras)` (`:140`) | `(*tuple(loras), *distilled_lora)` (`:151`) |
| `ic_lora.py` | `loras=tuple(loras)` (`:108`) | `loras=()` (`:119`) |
| `dfr_pipeline.py` | `(*user, *distilled)` on the one stage both phases share (`:212`, `:217`) | — |

The first two hand stage 1 the USER adapters and stage 2 those PLUS the
distilled one. That is a proper subset. It stayed invisible only because the
arity cap made `tuple(loras)` necessarily EMPTY: this engine's single
`lora_path` extra IS the `distilled_lora` those recipes demand, so `kNoAdapters`
on stage 1 was exact for a one-adapter load. **Lifting the cap makes the wrong
arm expressible**, and this engine cannot run it — it holds one resident DiT,
`Ltx2RebindDitLoras` takes a BOOLEAN, and no load extra says which of the N is
the distilled one.

**So the load refuses it by name** (`ltx2_video.cpp`, keyed on the phase scope
and not on `requires_distilled_lora`, because the scope is the thing that cannot
be represented). Three shipped recipes carry a `kNoAdapters` phase —
`a2vid_two_stage`, `ti2vid_two_stage`, `keyframe_interpolation` — and on those a
second adapter refuses. Every recipe whose phases all run fused, `dfr` included,
takes N. The enum stays two-valued because a third value would name a state
nothing could execute; the arm is listed under `## Owed` below.

What `ti2vid_two_stages_hq.py:154` against `:165` varies per stage is the
STRENGTH (0.25 at `:92-96`, 0.5 at `:97-101`), not the membership, and that is
#1144.

## 6. Risks

- **The second form is a branch that was previously unreachable.** It is now
  reachable, and a wrong rounding there is invisible to a token gate and to any
  tolerance. Mitigated by §3's golden, generated by EXECUTING the pinned module,
  and asserted as byte equality on the bf16 bit patterns.
- **The f32 intermediate.** The second form needs a `vt::Matmul` with an f32
  output where the first uses bf16. A reviewer should check that the f32 buffer
  is not carried into the first form, which would silently widen the landed and
  gated arm.
- **Reachability.** Lifting a refusal in `ltx2_lora.cpp` alone would land a
  capability no user can request, because the extras surface held one path. The
  transport change is what makes the row reachable, and §8's mutation is on the
  transport and not on the fuser.
- **The derived reader-anchor list** in `ltx2_video.cpp` moves when lines are
  inserted above it. Re-derived from the gate's own printed replacement.
- **The refusal runs AFTER the fusion it discards, and that is recorded rather
  than restructured.** `ResolveLoraSpecs` feeds `Ltx2LoadDitFromSafetensors`,
  the recipe resolves after that, and the `kNoAdapters` guard runs after both,
  so on real weights a two-adapter `a2vid_two_stage` load fuses a 21 B DiT twice
  and only then refuses. **Not introduced here**: the `requires_distilled_lora`
  refusal in the same function has had the same placement since it landed, and
  the guard has to read `im.recipe.phases`, which does not exist until the
  recipe resolves. Hoisting it would mean lifting recipe resolution above the
  DiT load, which is a change to the load order of every arm and belongs to a
  row that can gate it. The cost is wall time on a refused load, never a wrong
  render.

## 7. Gates

```sh
cmake --build build --target test_ltx2_lora test_ltx2_video test_ltx2_loader -j 4
./build/tests/test_ltx2_lora
./build/tests/test_ltx2_video -tc="*lora*"
./build/tests/test_ltx2_loader
python3 scripts/agent-integration.py --base origin/main
```

## 8. Evidence

Measured on this row's branch, CPU only. `cmake -S . -B <build> -G Ninja
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF` (no `CMAKE_BUILD_TYPE`, so no
`NDEBUG`), `-j 4`.

### 8.1 The golden, re-derived

`aggregate_lora_products` + `bf16_fuse_rule` imported from
`/home/mudler/_git/LTX-2/packages/ltx-core/src` at `fd4ded7f` (clean tree),
torch `2.11.0+cu130`, and run on the same bf16 inputs the case builds
(`Spread` reimplemented as the same Numerical Recipes LCG). The 120 bit
patterns it printed are byte-for-byte the `want` array in
`test_ltx2_lora.cpp` — re-derived independently after the first implementer's
session ended, not inherited.

### 8.2 The three-model probe

Re-run: 49 trials, model A matched on all 49, B failed 47 and C failed 45. §3
carries the table. On this row's own fixture B differs from upstream on 18 of
120 elements and C on 21 of 120; the plausible wrong port that reuses the FIRST
product form for every adapter differs on 26 of 120, which is the number
`test_ltx2_lora` asserts.

### 8.3 Red-before, three of them

| mutation | result |
|---|---|
| the fuser at its parent commit (both refusals in place) | `test_ltx2_lora` 15/17, the two new cases RED on "exactly ONE adapter" and on the `addmm_` refusal by name |
| model B substituted into the second form (`alpha * bf16(prod[i])`) | `test_ltx2_lora` 16/17, the golden RED at `mismatched == 18`, which is §8.2's own separation count arriving from the other side |
| the §5 guard disabled (`&& false`) | `test_ltx2_video` "a SECOND adapter refuses, because stage 1 cannot hold a SUBSET" RED at `REQUIRE_FALSE(message.empty())` — the load ACCEPTED two adapters on `a2vid_two_stage` and would have rendered |

Every mutation build exited 0 before its suite ran, so no red above is a build
failure wearing a test failure's clothes.

### 8.4 The reachability mutation

`ResolveLoraSpecs`'s loop changed to `for (int64_t index = 1; index <= 1;
++index)` — the `index > 1` arm deleted, which is the production call site for
everything this row adds. Re-run at the FINAL head, after the merge with
`origin/main`, because a mutation proved at a parent commit proves nothing about
what lands. Build exit 0.

- `test_ltx2_lora`: **17/17, 157 assertions, SUCCESS** — unchanged. A unit suite
  that builds three `Ltx2LoraAdapter`s by hand cannot see that no user can ask
  for three.
- `test_ltx2_video`: **RED** in three cases — "a SECOND IC-LoRA supplied through
  `lora_path_2` reaches the PIXELS" at `CHECK(differing > 0)`, two subcases of
  the indexed-refusal case, and "a SECOND adapter refuses, because stage 1
  cannot hold a SUBSET" at `REQUIRE_FALSE(message.empty())`, since a load that
  parses one adapter never reaches §5's guard. Entry point `LoadVideoEngine` ->
  `Generate`, no internal header.

That difference is the claim. The tree was restored byte-for-byte after each
mutation (`sha256sum` on both files, before and after).

### 8.5 What the goldens do NOT cover

- **Torch's GEMM reduction order at production shapes.** Model A is a strictly
  sequential f32 accumulation along `K`, which is what `vt::Matmul`'s CPU kernel
  does. Against the pinned module it is exact at every shape the golden and the
  probe reach. At full DiT shapes it is not: `M=2048 K=64 N=2048` differs on 42
  of 4194304 elements, `M=4096 K=128 N=1024` on 58 of 4194304, `M=1152 K=32
  N=4608` on 32 of 5308416. **This is not something this row introduced.** The
  landed and gated FIRST form has the same property on the same shapes — 25 and
  51 of 4194304 on the first two — because it is torch's blocked bf16 GEMM
  rounding differently from a sequential reduction, not a difference between the
  two product forms. Stated here because a reader of the byte-equality claim
  would otherwise read it as reaching further than it does.
- **A tensor targeted by more than three adapters.** The golden has three; the
  probe reaches four. Nothing gates 5+, and the loop is uniform past the first.
- **The device arm.** The fusion is host-side by construction (§9), so no
  golden here says anything about a fused weight after the device copy.
- **Real adapter files.** `WriteAdapter` and `WriteFixtureLora` build the
  safetensors; no published multi-adapter LTX-2.5 recipe was run.

### 8.6 The focused gate

`test_ltx2_lora` 17/17 (157 assertions), `test_ltx2_loader` 41/41 (64246),
`test_ltx2_video` 112/112 (4931), all exit 0, on the head merged with
`origin/main`. The video suite includes the two record cases this change
invalidated and repaired:

- **"every accepted load extra is READ by something"** counted 18 names against
  an inventory of 16. The indexed family is now in `served` with its reader
  (`ResolveLoraSpecs`) named, which is what that gate's own failure message asks
  for.
- **"the recorded reader anchors are the ones in the source"** derived
  `lora_path` and `lora_strength` to the SAME line, because the `..._1` by-name
  refusal named both tokens between `kKnownLoadExtras[]` and the real readers,
  and that gate defines a key's reader as the first mention after the array. The
  refusal and the indexed listing moved ABOVE the array (`RefuseLoraIndexOne`,
  `LoraIndexedListing`) so the gate keeps its meaning, and the anchors were then
  re-derived from the gate's own printed replacement rather than by arithmetic,
  TWICE: once before the merge with `origin/main` and again after it moved five
  of them. Two anchors are now 572/574 because the read itself moved into
  `ResolveLoraSpecs`, which is a real reader.

`scripts/agent-preflight.sh` exits 0 with every checker `ok` and two SKIPs that
need `--compile-commands`, which preflight does not supply.

## 9. Stop conditions

- `NEEDS_DECISION` rather than growing into A15 or A16.
- `NEEDS_DECISION` rather than inventing a per-phase adapter SUBSET that no
  upstream recipe asks for.
- Do not use the GPU. This row is CPU-only by construction: the fusion runs on
  the host before the device copy.

## Owed

- **N adapters on a recipe with a `kNoAdapters` phase.** `a2vid_two_stage`,
  `ti2vid_two_stage` and `keyframe_interpolation` need stage 1 to carry the USER
  adapters and not the distilled one (`a2vid_two_stage.py:107` against `:114`),
  which is a proper subset this engine cannot hold: one resident DiT,
  `Ltx2RebindDitLoras` is a boolean, and no load extra says which adapter is the
  distilled one. Refused by name at load (§5). Closing it needs a third
  `Ltx2PhaseLoraScope` value, a per-phase adapter selection in
  `Ltx2RebindDitLoras`, and a load extra that identifies the distilled adapter.

## Now

`ACTIVE` — N-adapter fusion implemented, gated against the executed pinned
module, and reachable through the `lora_path_<n>` load extras and repeated
`ltx2_gen --lora`. The one arm lifting the cap made expressible and this engine
cannot run — a second adapter on a recipe with a `kNoAdapters` phase — refuses by
name and is listed under `## Owed`.
