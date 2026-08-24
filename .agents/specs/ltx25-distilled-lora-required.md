# LTX-2.5 — the two two-stage arms that ran a distilled schedule and demanded nothing

Row `LTX25-DISTILLED-LORA-REQUIRED`.
Issue [#1445](https://github.com/mudler/vllm.cpp/issues/1445).
Owning row of record: `MODEL-DIFFUSION-LTX25`.

## 0. Honesty statement — what this row does and does not claim

This row sets one boolean on two recipes and makes one sentence of one refusal
message derived instead of literal. It does not add a check. The check has been
in `ltx2_video.cpp` since [#1117](https://github.com/mudler/vllm.cpp/issues/1117)
and it is keyed on the recipe, precisely so that the next recipe inherits it.
Two recipes did not carry the key.

It does NOT claim that a res2s or dfr render is correct once the adapter is
supplied. It is not: this engine fuses one load-time strength into every phase,
and upstream's HQ preset uses `0.25` on stage 1 and `0.5` on stage 2. That gap
is [#1144](https://github.com/mudler/vllm.cpp/issues/1144)'s and this row makes
it reachable on every res2s load rather than only on the ones that supplied an
adapter voluntarily. Section 5 says why that is still the right trade.

No GPU lease was taken and no real checkpoint was loaded. Every gate here runs
on the reduced fixture DiT.

## 1. The defect

`Ltx2PipelineRecipe::requires_distilled_lora` is the field that refuses a load
carrying no adapter. Three factories set it — `A2VidTwoStageRecipe`,
`Ti2VidTwoStageRecipe`, `KeyframeInterpolationRecipe`. `Res2sTwoStageRecipe` and
`DfrRecipe` did not, and both run distilled sigma schedules on a checkpoint that
is not the distilled one:

- `Res2sTwoStageRecipe` gives stage 2 `Stage2DistilledSigmas()` and re-noises to
  its first element, on a `kFull` checkpoint.
- `DfrRecipe` IS `DistilledTwoStageRecipe` with two phases renamed, so BOTH its
  stages take the distilled schedules, on a `kKeyframeSlotSft` checkpoint.

Run either with no adapter and the render finished. The clip came back at the
requested size, frame count and sample rate, produced by a distilled schedule on
weights that were never distilled. No pixel, RMS, windowed-energy or spectral
check in this suite can see that — which is why the red-first evidence in
section 6 shows a full successful render trace immediately before the assertion
that says the load should have been refused.

## 2. Upstream — what actually requires it, and where

**Oracle.** vLLM registers nothing LTX at the parity pin, so this lane's
reference is `Lightricks/LTX-2`, pinned at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` by row `ENG-UPSTREAM-LTX2-PIN`
([#1433](https://github.com/mudler/vllm.cpp/issues/1433), PR
[#1869](https://github.com/mudler/vllm.cpp/pull/1869), which owns that pin's
record and which this row does not duplicate). **Identity asserted, not
assumed:** every anchor below was read in a checkout at
`/home/mudler/_git/LTX-2` whose `git rev-parse HEAD` is that revision and whose
`git status --short` is empty.

**Upstream states the rule once, for the whole class.**
`packages/ltx-pipelines/CLAUDE.md:48`:

> Two-stage non-distilled pipelines require `distilled_lora` (applied to stage 2
> only in TI2Vid/A2Vid/Keyframe).

The first half is the requirement and it is unqualified. The parenthetical
scopes WHERE the adapter runs for three of the five, and the next two lines
scope the other two:

- `CLAUDE.md:49` — "HQ applies distilled LoRA to **both** stages with separate
  `distilled_lora_strength_stage_1` / `_stage_2` params."
- `CLAUDE.md:50-51` — "DFR also applies distilled LoRA to **both** stages
  (shared stage object for stage 1 / temporal rounds; stage 2 may add a detailing
  IC-LoRA via `with_loras`)."

So both arms are inside the requirement and both differ from the three gated
arms in the SCOPE, not in the requirement. Per recipe:

| | `res2s_two_stage` — `TI2VidTwoStagesHQPipeline` | `dfr` — `DFRPipeline` |
|---|---|---|
| `Model` column | `Full + distilled LoRA (both stages)` (`CLAUDE.md:22`) | `Keyframe-slot SFT + distilled LoRA (+ detailing IC-LoRA stage 2)` (`CLAUDE.md:25`) |
| constructor | `distilled_lora` positional, non-defaulted, `ti2vid_two_stages_hq.py:74` | `distilled_lora` positional, non-defaulted, `dfr_pipeline.py:173` |
| cannot be built without one | INDEXED at `:93` — an empty list raises `IndexError` | docstring `:157` "on a keyframe-slot-capable SFT base plus a distilled LoRA" |
| where it lands | `:154` stage 1, `:165` stage 2, one strength each (`:92-100`) | `:212` `stage_loras = (*self._user_loras, *self._distilled_lora)`, handed to the ONE shared `DiffusionStage` at `:217` |
| CLI | `--distilled-lora required=True` (`utils/args.py:1140-1155`) on `default_2_stage_arg_parser` (`:1123`), via `hq_2_stage_arg_parser` (`:1168`, calls at `:1172`), selected at `:346` | the same argument, `default_2_stage_arg_parser` selected directly at `dfr_pipeline.py:568` |
| filled from args | `:350-352` | `:595` |

**The answer is BOTH, and the issue's framing is confirmed rather than assumed.**
Section 4 records the pairs where the answer is NO, because "which recipes do
not" is the half a table can be wrong about silently.

**The decision #1445 asked for: dfr's stage-2 detailing IC-LoRA is NOT a second
requirement.** `detailing_lora` defaults to `None` (`dfr_pipeline.py:176`),
`--detailing-lora` carries `default=[]` and no `required` (`:569-577`), and
`CLAUDE.md:51` says stage 2 "may add" it. So ONE adapter is the complete
requirement for this arm, `requires_distilled_lora` is the right shape, and the
arity cap in `Ltx2ResolveLoraReferenceFactors` that refuses a second adapter is a
missing CAPABILITY rather than an unmet requirement of this flag. It stays owned
where it already is; this row does not lift it.

**The upstream doc anchors in this tree had drifted, and this row corrects the
four that were wrong.** Measured against the pin: `ltx2_pipeline.cpp` cited
`CLAUDE.md:26` for HQ (actual `:22`), `:24` for DFR (actual `:25`), `:25` for
`DistilledPipeline` (actual `:26`), and `:22` for `KeyframeInterpolationPipeline`
(actual `:24`); `test_ltx2_video.cpp` cited `:25` for `DistilledPipeline`. The
other six citations of that table in `src/` and `tests/` resolve correctly. Fixed
in flow rather than filed, because this row's own edit makes `dfr` cite `:25` —
which is the line `DistilledPipeline` was wrongly citing — and leaving that
collision in place would have two recipes claiming one row of upstream's table.

## 3. Design

**One field, two recipes.** `recipe.requires_distilled_lora = true` in
`Res2sTwoStageRecipe` and `DfrRecipe`, each beside a comment carrying the anchors
in section 2. Nothing else about either recipe changes. The phase SCOPE was
already right: `Ltx2PhaseLoraScope::kAllAdapters` is the default and neither
recipe overrides it on either phase, which is `CLAUDE.md:49` and `:50-51`
already mirrored — see the header comment on `Ltx2PhaseLoraScope`, which cites
all three lines and predates this row.

**One sentence of the refusal becomes derived.** The message ended:

> The adapter runs on the phases the recipe's `loras` scope names, which for
> these pipelines is stage 2 alone.

True of the only three arms that carried the flag. FALSE for both arms this row
adds. A refusal that names the wrong phase sends the reader to the wrong file, so
the sentence now lists the phase names the recipe actually scoped the adapter to,
read off `recipe.phases`. `ti2vid_two_stage` prints `'stage_2'`;
`res2s_two_stage` prints `'generate_lowres_hq', 'refine_hq'`; `dfr` prints
`'dfr_base', 'dfr_detail'`.

A recipe that requires an adapter and scopes it to no phase would be incoherent.
That is unreachable for every recipe this tree ships, and the message says so in
words rather than printing an empty list that reads like a formatting bug. The
two fields are set independently and nothing pairs them, which is the only reason
the branch exists.

**No new refusal site, and no new order.** The adapter check already sits before
the checkpoint-class check, deliberately: a two-stage load with no adapter fails
both, and the adapter message names a missing file the caller can go and fetch.
`dfr` is the arm where that matters most, because its `keyframe_slot_sft` base is
not published at all.

**Rejected: refusing `res2s_two_stage` outright over the per-stage strength.**
Upstream sets `0.25`/`0.5` and this engine has one strength. AGENTS.md's "refuse
an unimplemented arm by name" would read on that. It is not what this row does,
because the arm is shipped and gated, the strength gap is already owned by
[#1144](https://github.com/mudler/vllm.cpp/issues/1144) with its own record, and
refusing it here would delete a capability under the cover of a requirement fix.
It is named instead — in the recipe comment, in `docs/models/ltx-2-5.md`, in
`docs/FEATURES.md`, and in `## Owed` below.

## 4. Tests

Every case enters through `LoadVideoEngine`, the function `vllm_video_engine_load`
calls. A case that constructed the recipe by hand would prove the struct works
and never that anything reaches it.

| Case | File | What it fails on |
|---|---|---|
| `ltx2 res2s: the distilled-LoRA requirement refuses BY WHAT IS MISSING` | `tests/vllm/multimodal/test_ltx2_video.cpp` | a `res2s_two_stage` load with no `lora_path` is accepted |
| `ltx2 dfr: the distilled-LoRA requirement refuses BY WHAT IS MISSING` | same | a `dfr` load with no `lora_path` is accepted, or the CLASS refusal is read instead of the adapter one |
| `ltx2 distilled LoRA: the phase scope in the refusal is READ OFF the recipe` | same | the scope sentence is a literal — asserted by holding a one-phase arm and a two-phase arm against each other in one case |
| `ltx2 distilled LoRA: every resolving pair carries upstream's ADAPTER half` | `tests/vllm/models/test_ltx2_pipeline.cpp` | any (kind, version) pair carries the wrong adapter half, or the wrong phase count |

The fourth is the one that did not exist before this row and is the reason the
defect survived: three recipes set the flag and seven did not, and NOTHING
asserted that the seven were right not to. The two that were wrong sat in the
majority and looked like the rule. It is a cross product over ten kinds and four
versions with the same 28-pair count the checkpoint-class table case asserts,
restated rather than shared, because a helper that produced both numbers could be
wrong once and agree with itself twice.

**Collateral, and what it means.** Six existing cases loaded `res2s_two_stage` or
`dfr` with no adapter and now have to supply one. They do it through
`SupplyRequiredAdapter`, which reads `requires_distilled_lora` off the recipe
rather than matching a list of kinds — for the same reason `FixtureCheckpointClass`
is derived. A case that supplied an adapter the recipe did not require would be
quietly changing the weights it renders on. `WriteFixtureLora` moved to the top of
the file, unchanged, because three of those six cases sit above where it was
defined.

The `ltx2 checkpoint class: an UNDECLARED load refuses` arm table flips
`res2s_two_stage` and `dfr` to `needs_adapter = true`. Without that flip the case
would silently start gating the ADAPTER refusal instead of the CLASS one — the
same message, twice, for two arms, and the case's own assertions would fail
loudly rather than drift. They did fail; that is how it was found.

## 5. Risks

- **This refuses loads that succeed today, on two shipped and gated arms.** That
  is the change, not a side effect. It is the same trade
  `ltx25-checkpoint-class.md` made for nine kinds and the reasoning is that row's:
  the accepted load produces a plausible artifact that no output check can
  falsify, so acceptance is the failure.
- **`res2s_two_stage` now reaches #1144 on every load.** Before this row a caller
  could run it with no adapter and get a wrong regime; now they must supply one
  and get the right regime at the wrong strengths. The second is strictly closer
  to upstream and is visible in the docs; the first was invisible.
- **`dfr` cannot be fed at all**, since no `keyframe_slot_sft` base is published
  (`ltx25-checkpoint-class.md` `## Owed`). Its blast radius on a real user is
  therefore nil, and the fixture suite is where it is observable.
- **The scope sentence is now derived from a field that no upstream check
  validates.** If a future recipe sets `kNoAdapters` on the wrong phase, the
  message will faithfully report the wrong scope. The table case in section 4
  pins the phase count per kind, which is what bounds that.

## 6. Gates

Run in `/home/mudler/_git/wt-1445b`, base `1724be38e24dc8100ffc448b420d44e8fe1e5d28`,
Release, CPU-only, x86-64.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_SERVER=ON
ninja -C build ltx2-gen test_ltx2_video test_ltx2_pipeline
./build/tests/test_ltx2_video
./build/tests/test_ltx2_pipeline
scripts/agent-preflight.sh --staged
```

`ltx2-gen` is in the gate because this row edits its `--help` text, which is
product output rather than a comment.

**A configuration trap, recorded so the next reader does not attribute it to a
diff.** The focused loop above first ran with `-DVLLM_CPP_SERVER=OFF
-DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_METAL=OFF
-DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_TENSTORRENT=OFF`, which is fast and adequate
for the two suites. A `ninja -C build` over ALL targets in that configuration
fails to LINK `test_minimax_music3_e2e_real` on six undefined
`vllm::entrypoints::openai::ApiServer` symbols. That is `SERVER=OFF`, not this
change: the test links the server unconditionally. Reconfigure with `SERVER=ON`
before reading an all-targets build as a verdict on anything.

**RED first, captured verbatim.** With the tests written and the flag still
false, all three `test_ltx2_video` cases failed for the intended reason. The
res2s case failed AFTER a complete render trace — `denoise`, `decode.video`,
`decode.audio`, `artifacts.audio` — which is the defect in one screen:

```
/home/mudler/_git/wt-1445b/tests/vllm/multimodal/test_ltx2_video.cpp:11422:
  ERROR: a res2s_two_stage load with no distilled LoRA must be refused
[doctest] test cases: 1 | 0 passed | 1 failed | 104 skipped
```

```
/home/mudler/_git/wt-1445b/tests/vllm/multimodal/test_ltx2_video.cpp:11476:
  ERROR: a dfr load with no distilled LoRA must be refused
[doctest] test cases: 1 | 0 passed | 1 failed | 104 skipped
```

The scope case failed by printing the literal it exists to remove:

```
ERROR: CHECK( stage2_only.find("stage_2") != std::string::npos ) is NOT correct!
logged: ... The adapter runs on the phases the recipe's `loras` scope names,
        which for these pipelines is stage 2 alone.
```

**GREEN after,** on the declared configuration above: `test_ltx2_video`
105 cases / 4339 assertions, `test_ltx2_pipeline` 60 / 3475. `ltx2-gen` links
and its `--help` prints the new requirement.

**The `test_ltx2_video` assertion count is NOT REPRODUCIBLE, and an earlier
draft of this line got the reason wrong.** It first said the count was keyed to
the build configuration — 4337 under `-DVLLM_CPP_SERVER=OFF`, 4339 with the
server on. The fresh review measured four consecutive runs of ONE binary and got
4338, 4339, 4340, 4343; a fifth, sixth and seventh run of a different single
binary here (`md5 b6fbd5c8da73e1389a01c40852ad7b7f`) gave 4341, 4342, 4337. So
the number moves run to run on a fixed binary, and comparing two configurations
to explain a difference one binary produces on its own is a wasted A/B.

**The case count is the stable figure: 105, always, with 0 failed.** Read that.
Never read the assertion count as a measurement of anything, and never read a
change in it as a regression. Pre-existing, outside this row, and filed under
`## Owed`.

**Mutations — six, all DETECTED.** Each anchor was asserted UNIQUE before it was
applied, and the tree was restored and its sha256 compared byte-for-byte after
each one.

| # | Mutation | Result |
|---|---|---|
| M1 | delete `requires_distilled_lora = true` from `Res2sTwoStageRecipe` | res2s case and scope case RED |
| M1b | same | the `ADAPTER half` table case RED — which is also the proof that case is red on `origin/main` |
| M2 | delete it from `DfrRecipe` | dfr case RED |
| M2b | same | table case RED |
| M3 | restore the old literal scope sentence | scope case, res2s case and dfr case all RED |
| M4 | **REACHABILITY** — disable the production call site in `ltx2_video.cpp` | all three new cases RED, and the two pre-existing `ti2vid` / `keyframe` cases RED with them |

M4 is the one that matters for `.agents/reachability.md`: the gate measures a
capability reached through the loader, not a class.

## Owed

- [#1144](https://github.com/mudler/vllm.cpp/issues/1144) — the per-stage
  distilled-LoRA STRENGTH. Upstream's HQ preset is `0.25` on stage 1 and `0.5` on
  stage 2; this engine fuses one load-time strength into every phase. This row
  makes that path mandatory on `res2s_two_stage` rather than optional, and does
  not close it. Named in `docs/models/ltx-2-5.md` and in the recipe comment.

  **Two files, and the fresh review caught the conflation.** The SPLIT — one
  adapter into two `LoraPathStrengthAndSDOps` objects, one per stage — is
  `ti2vid_two_stages_hq.py:92-100`. The two VALUES are the CLI defaults
  `utils/args.py:1176` and `:1182`. Neither literal appears in
  `ti2vid_two_stages_hq.py`, and an earlier draft of the recipe comment sourced
  them there. The load-bearing claim was always the split; only the numbers were
  attributed to the wrong file.
- **`dfr`'s stage-2 detailing IC-LoRA is a second, OPTIONAL adapter this engine
  cannot carry.** `Ltx2ResolveLoraReferenceFactors` refuses more than one adapter
  by arity. Upstream makes it optional (`dfr_pipeline.py:176`, `:569-577`), so it
  is not owed by this flag, and this row does not lift the cap. Recorded here so
  the decision has a home; the capability stays where the arity refusal already
  records it.

- [#1885](https://github.com/mudler/vllm.cpp/issues/1885) — **`test_ltx2_video`
  reports a different assertion count on every run of the same binary.** Seven
  runs across two binaries at this commit gave 4337, 4338, 4339, 4340, 4341,
  4342 and 4343, always at 105 cases and 0 failed. Filed by this row because
  this row's own §6 first mis-explained it as a build-configuration difference,
  and because a count comparison across a diff is a thing this repository's
  records actually do. NOT fixed here: the varying cases are pre-existing and
  this row touches none of them, and isolating them needs a bisection over 105
  cases with its own row.

## Now

`ACTIVE`. The spec is committed before the implementation on branch
`row/LTX25-DISTILLED-LORA-REQUIRED`, which is the commit order that proves
spec-first. The row reaches `DONE` when a fresh reviewer returns `PASS` on the
immutable head and the operator reruns the gate.
