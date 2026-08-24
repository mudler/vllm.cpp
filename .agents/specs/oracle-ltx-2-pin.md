# SPEC — pinning Lightricks/LTX-2, the LTX-2.5 lane's own reference

**Row:** `ENG-UPSTREAM-LTX2-PIN`
**Issue:** [#1433](https://github.com/mudler/vllm.cpp/issues/1433)
**State:** `READY` — record and policy change, no product code.

## Scope

Admit `Lightricks/LTX-2` to the oracle registry and pin it, so that the
revision the whole LTX-2.5 lane already reads its upstream truth from is a
recorded oracle rather than a string repeated in prose.

In scope: the AGENTS.md admissible-oracle table row, the
[`.agents/oracles/ltx-2.md`](../oracles/ltx-2.md) record, the identity assertion
that establishes which revision was read, and an honest gateability verdict with
the issue that owes the measurement.

Out of scope: running the oracle (no GPU lease was authorised for this row, and
the run is what `gateable` is owed); advancing or re-reading any anchor that
already cites `fd4ded7f`; changing `scripts/check-oracle-pins.py`, whose
semantics this row deliberately leaves alone; and re-stating anything AGENTS.md
§"When vLLM has no implementation" or
[`.agents/oracles/README.md`](../oracles/README.md) already says.

**Why this is a policy change and not an in-flow fix.** The registry table lives
in AGENTS.md, and `check-oracle-pins.py` reads that table as one of the two sides
it holds equal. Adding a row changes what the protocol admits, so it takes a
spec and a fresh review rather than riding in another change.

## Upstream chain

`Lightricks/LTX-2` — <https://github.com/Lightricks/LTX-2>, the model author's own
runtime, a **third** repository that is neither `vllm-project/vllm` nor
`vllm-project/vllm-omni`. Two packages carry everything this lane reads:
`packages/ltx-core/src/ltx_core/` (the architecture) and
`packages/ltx-pipelines/src/ltx_pipelines/` (the recipes, sigma schedules and
defaults). Both declare `version = 1.2.0` at the pin.

**Identity, asserted rather than assumed.** The revision recorded here was read
from a local clone at `/home/mudler/_git/LTX-2` whose `origin` is
`https://github.com/Lightricks/LTX-2.git`, whose worktree is clean, and whose
`HEAD` is `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` — the merge of upstream
`Lightricks/LTX-2#273` (`patch/pin-transformers-below-5-15`), authored
2026-08-11. The repository carries **no tags**, so the package version string is
the only human name the revision has, and `pin_label` uses it.

One anchor was re-derived from that clone rather than transcribed from our
records: `git show fd4ded7f:packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py`
carries `DISTILLED_SIGMA_VALUES` at line 17, which is the value
[`porting-inventory.md`](../porting-inventory.md) cites for the distilled sigma
schedule.

## Our baseline

The lane already treats this repository as its oracle in everything but the
record. `fd4ded7f` appears in ten `.agents/specs/ltx25-*.md` files, in
[`porting-inventory.md`](../porting-inventory.md), in
[`kernel-matrix.md`](../kernel-matrix.md), in five `scripts/gen-ltx2-*.py`
golden generators and in fifteen `src/vllm/model_executor/models/ltx2_*.cpp`
translation units. `.agents/specs/ltx-2-5.md:8` names it as the architecture
upstream outright.

What did not exist before this row: any `.agents/oracles/<id>.md` file for it,
any row in the AGENTS.md table, and therefore any pin.
`python3 scripts/check-oracle-pins.py` reported `oracle-pins ok (9 oracles
pinned)` while the lane's actual reference was absent from both sides it
compares. That is not a checker defect — the checker holds the table and the
directory equal, and an oracle in neither is in agreement with itself.

## Gateability — measured, and the answer is no

The bar is AGENTS.md's: the oracle must demonstrably build and run **the model**,
and constructing a config proves nothing. It has never run here.

Thirteen tracked scripts import and execute upstream `ltx_core` /
`ltx_pipelines` code (`grep -rn "import ltx_core\|from ltx_core\|import
ltx_pipelines\|from ltx_pipelines" --include="*.py"`). Every one of them runs
individual upstream **modules** at reduced dimensions on **synthetic PRNG
weights**, or reads constants and safetensors headers. The committed goldens say
so themselves — `tests/vllm/models/ltx2_vae_goldens.inc:3-6`: "Weights and inputs
come from the shared deterministic stream, so no weight byte is checked in", and
the same sentence appears in `ltx2_goldens.inc`, `ltx2_pipeline_goldens.inc`,
`ltx2_text_goldens.inc`, `ltx2_tiling_goldens.inc` and
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc`.

Two scripts touch real checkpoint bytes, and neither runs the model:
`scripts/measure-ltx2-prompt-adaln.py:124-133` forwards one `AdaLayerNormSingle`
module plus the per-block tables out of a 21 B DiT, and
`scripts/measure-ltx2-keyframes-meta.py:156,203` builds the model on the **meta
device** and runs upstream's loader with no forward pass anywhere in the file.
Neither writes a committed artifact.

Negatives, with the searches, so none rests on one grep: `ls tools/oracle/` holds
`music3_oracle.py` and `README.md`; `grep -rn -i ltx tools/` returns no hit across
the 40 files there; `ls tests/parity/goldens/ | grep -i ltx` matches nothing among
105 entries. `.agents/specs/ltx-2-5.md` §7.1 states the method in its own words —
upstream's modules are "executed at reduced dimensions on CPU as the oracle" — and
§7.0 records "BINDING-ORACLE PARITY IS PENDING FOR EVERY BRICK LANDED SO FAR".

So `gateable = no`, and [#1864](https://github.com/mudler/vllm.cpp/issues/1864)
owes the measurement. **This is the correct verdict, not a shortfall**: it turns
an absence into visible debt. No GPU lease was authorised for this row, and a
`yes` could therefore only have been asserted.

One finding is worth carrying into that issue rather than leaving in a report.
`.agents/oracles/diffusers.md` records that `tools/oracle/music3_oracle.py`
asserts the installed revision **before it loads a weight**. LTX has no
counterpart: where the SHA is asserted
(`scripts/gen-ltx2-res2s-goldens.py:397-402`, a hard `SystemExit`) no weight is
loaded, and where weights are loaded
(`scripts/measure-ltx2-prompt-adaln.py:60-66`) only the interpreter *path* is
asserted. `.agents/specs/ltx-2-5.md` §7.0(b) records why that matters: a decoy
`ltx_core` once produced byte-identical goldens and exited 0.

Checked and NOT filed as a defect, because reading further disproved it:
`scripts/gen-ltx2-tiling-goldens.py:702-704` only *warns* on a pin mismatch
instead of exiting, which looks like a hole and is not one — it emits the
resolved revision as `kLtx2TilingUpstreamRevision`, and
`tests/vllm/models/test_ltx2_tiling.cpp:88,392-393` fails on anything but the
pin. The warn is the first half of a two-step refusal.

## Port map

Three files, no product code.

| File | Change |
|---|---|
| `AGENTS.md` | one row inside the `oracle-registry` markers, id `ltx-2` |
| `.agents/oracles/ltx-2.md` | new record with the one `oracle-pin` block |
| `.agents/engine-matrix.md` | this row, its section count, and the total |

`scripts/check-agent-record.py`'s `ENGINE_ROWS` moves with the matrix, which is
the recorded mark for a counted record and not a semantic assertion.

## Why a separate oracle rather than vLLM-Omni's scope

vLLM at the parity pin `5559679229` registers **nothing** LTX. Searched, in that
checkout at that revision: `git grep -inE "\bltx" 5559679229 -- '*.py' '*.md'
'*.yaml' '*.yml' '*.txt' '*.json'` returns no line, and
`git grep -ilE "lightricks" 5559679229` returns no file at all. A whole-tree
case-insensitive `ltx` search returns eight paths, all of them binary — seven
PNG/SVG documentation assets and `swagger-ui-bundle.js` — so they are incidental
byte matches and not registrations. `vllm/model_executor/models/registry.py`
exists at that revision and is inside the searched set.

vLLM-Omni **does** register LTX2 — `vllm_omni/diffusion/registry.py:69-87` at
`a4ea67a2` names `LTX2Pipeline`, `LTX2DistilledPipeline`, `LTX2T2VDMD2Pipeline`
and `LTX2I2VDMD2Pipeline` — and it stays the preferred reference wherever it
reaches. It does not reach generation 2.5: its `_PIPELINE_RECIPES` stop at 2.3
(`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:161-166`), and 2.5 is open
upstream at [vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066).

The two references also **disagree**, which is the sharpest argument that one
cannot stand in for the other: Lightricks' default negative prompt carries five
leading tags vLLM-Omni's lacks, so this port keeps both strings, gives each row
its own source's, and gates the disagreement as a value
(`kLtx2NegativePromptsAgree`) rather than resolving it by preference. A scope
line that folded Lightricks into `vllm-omni` would make that gated disagreement
unstatable.

This is the shape [`diffusers`](../oracles/diffusers.md) already has: prefer
vLLM-Omni where it implements the pipeline, reach the other where it does not,
and record which of the two a stage was gated against.

## Tests to port

None. There is no upstream test to port: the change adds a record, and the
executable statement about a record is the checker that reads it.

`scripts/check-oracle-pins.py` already enforces every field-level rule this
record must satisfy, in both directions, and it does so **without being
modified**: the new file must carry exactly one well-formed block, the id must
match the filename stem, `gateable = no` must name the owing issue as `#N`, and
the AGENTS.md table and the directory must name the same set of ids. Landing the
file without the table row, or the table row without the file, is refused by
`check_registry`.

Deliberately NOT done here: teaching the checker to notice an oracle that is in
neither side. That is a semantic checker change and owes its own row, spec,
red-before test and green-after evidence. This row would be its first fixture,
not its author.

## Gates

```sh
python3 scripts/check-oracle-pins.py            # both directions, table vs directory
python3 scripts/check-oracle-pins.py --self-test
python3 scripts/check-agent-record.py
python3 -m pytest tests/scripts/test_check_oracle_pins.py tests/scripts/test_agent_record.py
scripts/agent-preflight.sh --staged
```

The first is the binding one, and it is a real transition rather than a
restatement: it reports **9 oracles pinned** on `origin/main` and **10** on this
head. That it fails on either half alone is proved by mutation, not assumed —
each was applied to the working tree, measured, and reverted:

| Mutation | Result |
|---|---|
| remove `.agents/oracles/ltx-2.md`, keep the table row | rc=1, `ltx-2: admitted by the AGENTS.md table but has no .agents/oracles/ record, so it has no pin` |
| remove the table row, keep the file | rc=1, `ltx-2: pinned in .agents/oracles/ but absent from the AGENTS.md table` |
| neither | rc=0, `oracle-pins ok (10 oracles pinned)` |

## Dependencies

None blocking. Related, and deliberately not merged into this row:

- [#633](https://github.com/mudler/vllm.cpp/issues/633) owes the **vLLM-Omni**
  pin. It stays `UNPINNED`. Pinning Lightricks does not pay that debt, and this
  lane is unpinned twice over until both land.
- [#1012](https://github.com/mudler/vllm.cpp/issues/1012) records that
  `diffusers` at its recorded pin also implements LTX-2.5. That is a scope
  correction on `diffusers.md`, on another keyed record, and belongs to whoever
  owns that row.
- [#1854](https://github.com/mudler/vllm.cpp/issues/1854) states the same
  absence from the other end — "artefact-freedom needs an absolute reference
  render from an oracle that runs this pipeline, which `.agents/oracles/` does
  not have". This row gives that oracle a name and a pin. It does not give it a
  run.

## Work breakdown

- **W1 (this change).** The registry row, the record, the identity assertion,
  `gateable = no` with its owing issue, and the matrix row.
- **W2 (owed, GPU-blocked).** Run `ltx_core` / `ltx_pipelines` at `fd4ded7f` on
  real LTX-2.5 weights, produce a committed reference render and manifest that
  records the revision and environment, assert the installed revision before any
  weight loads the way `tools/oracle/music3_oracle.py` does, and only then move
  `gateable` to `yes` with the manifest path as `evidence`.

## Risks/decisions

**`gateable = no` is the finding, not a shortfall of this row.** AGENTS.md sets
the bar at demonstrably building and running the model, and says constructing a
config proves nothing. No GPU lease was authorised for this row, so a `yes` here
could only have been asserted, and an asserted gateability is exactly what the
registry exists to stop. Recording `no` with an owing issue is what makes the
ungateable lane visible debt.

**The pin is not network-verified, by design.** `check-oracle-pins.py` is
deliberately network-free, so a 40-hex string passes shape and fails review. The
defence here is that the revision was read out of a clone whose remote, cleanliness
and `HEAD` are all stated above, and that the anchor re-derived from it matches
what our records cite. No claim is made about what `origin/main` holds upstream
today.

**The registry now names two oracles for one lane, and that is deliberate.**
`vllm-omni` and `ltx-2` both answer for LTX-2.5, at different reaches. The rule
that keeps this from becoming preference-shopping is the one `diffusers.md`
already states and this record repeats: prefer vLLM-Omni where it implements the
behavior, and record which of the two a stage was gated against.

## Owed

- The `gateable` measurement (W2), tracked by
  [#1864](https://github.com/mudler/vllm.cpp/issues/1864), which is the issue
  `.agents/oracles/ltx-2.md` names in its `evidence` field.

## Now

`READY`. W1 is this pull request. W2 is unclaimed and needs a GPU lease.
