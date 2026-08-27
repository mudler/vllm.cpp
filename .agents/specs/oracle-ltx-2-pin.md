# SPEC — pinning Lightricks/LTX-2, the LTX-2.5 lane's own reference

**Row:** `ENG-UPSTREAM-LTX2-PIN`
**Issue:** [#1433](https://github.com/mudler/vllm.cpp/issues/1433)
**State:** `READY` — W1 pinned the oracle; W2 ran it on real weights and moved
`gateable` to `yes`. The matrix row stays `READY` until this change lands, because
`check-agent-record` requires a `DONE` row to name its **closing commit** in the
owner column, and that hexadecimal does not exist before the merge.

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
record. `git grep -l fd4ded7f`, measured at this head rather than quoted from the
issue, returns **30** files matching `.agents/specs/ltx25-*.md`, **5** under
`scripts/` — four `gen-ltx2-*.py` golden generators and
`probe_ltx2_tiling_layout.py` — and **15** under `src/`, twelve of them named
`ltx2_*` — three named subsets of the **97** files in the tree that carry the
revision, not a partition of them. Widening to all of `.agents/specs/` gives 32,
and the two it adds are
[`ltx-2-5.md`](ltx-2-5.md) and this file, so that number partly counts itself.
[`porting-inventory.md`](../porting-inventory.md) (5 hits) and
[`kernel-matrix.md`](../kernel-matrix.md) (1) carry the revision too and are in
NEITHER glob, because they sit one directory up at `.agents/`.
`.agents/specs/ltx-2-5.md:8` names it as the architecture upstream outright.

The first draft of this spec said "ten", "five" and "fifteen", transcribed from
the `#1433` index row instead of re-derived, and a fresh review measured all
three wrong. Recording that here rather than only fixing it: a number quoted
often gets treated as measured, and this file is where the measurement now
lives.

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
come from the shared deterministic stream, so no weight byte is checked in".
`ltx2_pipeline_goldens.inc` repeats that sentence verbatim.
`ltx2_goldens.inc` and `ltx2_text_goldens.inc` carry the operative clause in a
different sentence — "the MATH is gated exactly here and no weight byte is
checked in" — and `ltx2_tiling_goldens.inc:6-8` and
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc:4-6` say it in their own
words. Of the six this record names: one clause in all of them, two identical
sentences. That is a sample and not the population — other LTX-2 artifacts carry
the clause too, and a normalised search of every tracked file finds the verbatim
sentence in exactly two `.inc` files. An earlier draft sorted
four into "verbatim" and a fresh review measured two of them wrong; the same
oversorted claim is in this row's `.agents/issue-index.md` entry, which is
append-only and therefore cannot be corrected there. This is the corrected
statement, and the substance never moved: no file in the set checks in a weight
byte.

Two of those thirteen touch real checkpoint bytes, and neither runs the model:
`scripts/measure-ltx2-prompt-adaln.py:126-140` forwards one `AdaLayerNormSingle`
module plus the per-block tables out of a 21 B DiT, and
`scripts/measure-ltx2-keyframes-meta.py:156,203` builds the model on the **meta
device** and runs upstream's loader with no forward pass anywhere in the file.
Neither writes a committed artifact.

Negatives, with the searches and with denominators that were counted rather than
estimated, so none rests on one grep: `ls tools/oracle/` holds `music3_oracle.py`
and `README.md`; `grep -rn -i ltx tools/` returns no hit across the **72** files
`git ls-files tools` reports; and none of the **103** entries under
`tests/parity/goldens/` matches `ltx` case-insensitively.

Two scripts outside those thirteen do read real checkpoint bytes —
`scripts/gen-ltx2-quant-goldens.py` reads weight rows out of the FP8 checkpoint
and does commit an artifact, and `scripts/gen-ltx2-prompt-tokens-goldens.py`
pulls `tokenizer_json` out of one. Neither changes the verdict, and naming them
is what makes the verdict survive the obvious falsification attempt: neither
executes `ltx_core` or `ltx_pipelines` at all. They use torchao and raw
safetensors, so they are this project reading a checkpoint, not the oracle
running the model. `.agents/specs/ltx-2-5.md` §7.1 states the method in its own words —
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
case-insensitive `ltx` search returns eight paths — seven PNG/SVG documentation
assets and the vendored minified `swagger-ui-bundle.js`, which git treats as text
rather than binary — so they are incidental byte matches and not
registrations. `vllm/model_executor/models/registry.py`
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

None from upstream: the change adds a record, and the executable statement about
a record is the checker that reads it.

Three cases are added here, and they exist because two **pinned counts** move
with this row. `scripts/check-pr-size.py`'s `governance_checker` contract refuses
a checker edit whose only artifact is the constant, and it refused this row's
first push by name. Each case was proved red before it was green, by mutating the
thing it pins and restoring the tree:

| Case | Mutation that reds it |
|---|---|
| `tests/scripts/test_agent_record.py::test_ltx2_pin_row_is_inside_the_engine_ratchet` | rename the id CONSISTENTLY in the matrix and in this spec. `check-agent-record.py` stays `rc=0` at `ENGINE=171`, because a count cannot see a rename, and this case is the only thing in the file that reds: `AssertionError: 0 != 1 : ENG-UPSTREAM-LTX2-PIN`. Deleting the matrix row instead is a WEAKER probe and was the first draft's claim here: it reds through `setUpClass` with `.agents/engine-matrix.md: 170 engine rows; expected 171`, which errors all 28 cases in the class rather than this one |
| `tests/scripts/test_check_gate_commands.py::test_dropping_the_ltx2_pin_row_from_the_pin_breaks_it` | drop the id from `RUNNABLE_BASELINE`. It reds at `test_check_gate_commands.py:373` on `assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))`, and its MESSAGE names no id at all — it prints a 41-element set against itself. It is not a sole detector either: eleven pre-existing cases red on the same drop, `test_the_baseline_matches_the_shipped_record` among them. What it adds is its third assertion at `:376`, `runnable - reduced == {"ENG-UPSTREAM-LTX2-PIN"}`, which ties the audit's runnable set to this id. Under the drop the id does reach the output, but only through pytest's source echo of `:372`, never through the assertion message |
| `tests/scripts/test_check_gate_commands.py::test_the_ltx2_pin_row_is_credited_for_real_commands` | same drop, plus it reads the `## Gates` section below and fails if it stops naming a checker that can fail. This is the case whose message NAMES the row, at `:390`: `'ENG-UPSTREAM-LTX2-PIN' not found in frozenset(...)`. It is also a sole detector — strip `python3 scripts/check-oracle-pins.py` from that block and it is the only case that reds, across all three suites and not merely this file |

The first names **both** upstream-pin rows, not just this one. They are about
different repositories and both answer for LTX-2.5, so folding either into the
other leaves the matrix internally consistent and silently retires a pin — the
one state a count cannot see.

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
python3 -m pytest tests/scripts/test_check_oracle_pins.py \
                 tests/scripts/test_agent_record.py \
                 tests/scripts/test_check_gate_commands.py   # 185 cases
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
- **W2 (landed 2026-08-27).** Run `ltx_core` / `ltx_pipelines` at `fd4ded7f` on
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

A fresh review mutated this rather than reading it, and the result is worth
stating plainly: flip one hex digit of `pin` and `check-oracle-pins.py`,
`check-agent-record.py`, `check-gate-commands.py` and all 185 cases of
`test_check_oracle_pins.py` (24), `test_agent_record.py` (110) and
`test_check_gate_commands.py` (51) stay green.
The FIVE C++ suites that assert the revision literal —
`test_ltx2_tiling.cpp`, `test_ltx2_pipeline.cpp`, `test_ltx2_dfr.cpp`,
`test_ltx2_vae.cpp` and `test_ltx2_image_cond.cpp` — do not close the hole:
each compares a constant emitted into its goldens against one hardcoded in the
test file, none reads `.agents/oracles/`, and all need a build. A sixth file,
`test_ltx2_lora.cpp:9`, carries the literal in a header comment and asserts
nothing, so it is not one of them.
The value is held by re-derivation and review, not by a gate, and the record now
says so where the value is.

**The registry now names two oracles for one lane, and that is deliberate.**
`vllm-omni` and `ltx-2` both answer for LTX-2.5, at different reaches. The rule
that keeps this from becoming preference-shopping is the one `diffusers.md`
already states and this record repeats: prefer vLLM-Omni where it implements the
behavior, and record which of the two a stage was gated against.

## Owed

- ~~The `gateable` measurement (W2)~~, tracked by
  [#1864](https://github.com/mudler/vllm.cpp/issues/1864). **PAID on 2026-08-27**
  — see `## Outcome`. The four paragraphs below are kept because each records a
  blocker this spec asserted and then measured FALSE by satisfying it, and a
  reader who only sees the answer re-books the hardware. Read them as history.
  **When first written: still owed, and the blocker this spec recorded was
  wrong.** A `dgx:gpu0` lease was held on
  2026-08-25 and no render followed, because the wall is not hardware: the bf16
  Gemma-4 text tower is absent locally, it is a 24.46 GiB gated download this
  box is already granted access to (401 anonymous, 302 authenticated), and no
  large-download authority is recorded for this row. The four other slots
  `ti2vid_one_stage` needs are already on the NAS, the local torchao Gemma
  cannot substitute (upstream reads no torchao tensor at this pin), and no
  precomputed-embeddings route bypasses the tower for the pipelines that
  matter. The sizes, hashes, HTTP statuses and the upstream `file:line`
  citations are in `.agents/oracles/ltx-2.md` under "A lease was taken on
  2026-08-25". What W2 needs first is a decision on that download, not a
  device.

  **Both of those blockers are now cleared and W2 is still owed** (2026-08-25,
  second attempt). The download was authorised and completed: 26,263,858,182
  bytes on the NAS, sha256 `ef7243612f...61d1` measured off the bytes on disk
  and agreeing with the advertised `x-linked-etag`, so the checkpoint set
  `ti2vid_one_stage` needs is complete for the first time. A lease was held
  again and the render still did not run, for a third reason that is neither
  hardware nor the download: the job harness wedged at `setup.sh:5` on
  `HB=$(heartbeat setup)`, a command substitution that never closes because the
  heartbeat subshell it backgrounds holds the capture pipe open. It burned
  2h37m of `dgx:gpu0` and wrote a zero-byte log. The reproduction, the
  one-redirect fix and the costs that stay UNMEASURED are in
  `.agents/oracles/ltx-2.md` under "2026-08-25, second attempt".

  **2026-08-26, attempts three through five: three more defects, none of them in
  the model, the hardware or the checkpoint.** `torchvision` was absent and
  `transformers` hid the fact behind a class name; the install that fixed it was
  guarded by `if ! python -c 'import torch'` and never ran, because the worker's
  `/tmp` survived from the previous job; and the render then reached the Gemma-4
  text tower on real weights and died twenty seconds in, inside Triton's runtime,
  because the worker has no CPython headers and Triton JIT-builds a `cuda_utils`
  extension with `gcc` before it can launch a kernel. The full traceback, the
  `cc_cmd`, and why the compiler's own message never reached any log are in
  `.agents/oracles/ltx-2.md` under "2026-08-26, attempts three, four and five".

  Two rules came out of those three and are now enforced by the harness rather
  than remembered. **Every install is unconditional**, because an install guarded
  by a different package's state guarantees nothing on a worker whose `/tmp` is
  indeterminate between jobs. And **every dependency the render needs is proved
  in the setup step**, where a failure costs seconds, rather than discovered in
  the render step, where it costs 65.3 GiB of staged checkpoints and a 22 B model load: the
  import gate exits 93 and the Triton JIT gate exits 94 by compiling a real
  kernel, not by importing Triton, since the extension is built lazily on first
  launch.

- **The matrix `DONE` transition for this row.** `check-agent-record` requires a
  `DONE` row in `.agents/engine-matrix.md` to carry an exact local code anchor,
  an exact parity-ledger link, and an owner column holding the **hexadecimal
  closing commit**. That commit is the merge of the change that landed W2, so
  the transition cannot ride in it. Owned by this row, and it is a record edit
  rather than work: nothing measurable is outstanding.

- **[#2055](https://github.com/mudler/vllm.cpp/issues/2055) — the identity assert
  does not reach the render subprocess.** `tools/oracle/ltx2_oracle.py` checks the
  revision and the resolved `ltx_*` origins in the parent, then renders with
  `python -m` in a child whose `sys.path[0]` is the CWD, which the parent never
  inspected. A fresh reviewer ran a decoy `ltx_pipelines` from the CWD and the
  parent still printed `IDENTITY_OK`. The 2026-08-27 render is unaffected
  (`render.sh` issues no `cd`; `/workspace/ltx2-oracle/` holds no `ltx_*`
  package), so this is an over-claimed guarantee rather than a wrong measurement.
  Not fixed in flow because the script's sha256 is the provenance the `gateable`
  verdict rests on, and editing it would break that equality for a hardening that
  changed no result.

- **The 26 uncommitted artefacts are recorded and not recomputed.**
  `tests/scripts/test_ltx2_oracle_goldens.py` recomputes the two digests that are
  committed, and states the other 26 as a record for whoever fetches the frames
  from `/workspace/ltx2-oracle/out/`. Nothing gates a fetched copy of them. That
  is deliberate — 4.6 MB of PPM is not worth committing for one geometry — but it
  is a gap and not a design, and a second render at another geometry should
  settle whether the frames belong in the tree at all.

## Now

`READY`, with W2's work complete and in this change.
`.agents/oracles/ltx-2.md` reads `gateable = yes`, and its `evidence` is
`tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json` — a path the pin
checker requires to exist, produced by a run and not transcribed.

**One obligation is left, and it cannot be met in this change.**
`check-agent-record` requires a `DONE` row in `.agents/engine-matrix.md` to carry
an exact local code anchor, an exact parity-ledger link, and an owner column that
is the **hexadecimal closing commit**. That commit is the merge of this pull
request, so flipping the row here would mean writing a SHA that does not yet
exist. The row therefore stays `READY` in this change and the transition is owed
to the follow-up that can name the commit. The `gateable` verdict does not wait
on that: it rests on the manifest, which is in this change.

## Outcome

**What was measured.** Upstream `Lightricks/LTX-2` at `fd4ded7f` rendered 25
frames at 320x192 plus 1.02 s of stereo 48 kHz audio from the four real LTX-2.5
bf16 checkpoints, on `dgx:gpu0` (GB10, capability 12.1, torch 2.13.0+cu130), in
93.8 s of render inside 243.7 s of script inside **5m31s of lease**. All four
checkpoint digests are now derived a **second** time, on another host from
another copy, and all four agree to 64 hex characters with the values already on
`main`. None of them is new; corroboration is what the run adds. The gap
`docs/models/ltx-2-5.md` records — that `792a2bad...` "has never been compared
against the published artifact" — is untouched, because hashing our own copy
twice is not that comparison.

**Six attempts, and the five that failed were the deliverable's real cost.** No
attempt failed in the model, on the hardware, or on a checkpoint. They failed on
a heartbeat that held a command substitution open, a `cp` over a `soft` CIFS
mount whose failure the script ignored, an absent `torchvision` that
`transformers` reported as a missing CLASS, an install guarded by a *different*
package's absence on a worker whose `/tmp` is indeterminate between jobs, and
finally on absent CPython headers, because the Gemma-4 RoPE step reaches a Triton
kernel and Triton JIT-builds a CPython extension with `gcc` before it can launch
one. `.agents/oracles/ltx-2.md` carries five of the six with their reproductions.
The sixth is the `cp` — its evidence is the worker log line
`cp: error reading '...transformer-bf16.safetensors': Resource temporarily
unavailable` in `run-20260826T070001.render.log`, and the fix, a `.part` file
renamed only on a size match, is in the job harness rather than in this tree.

**Why the diagnosis took five attempts and one lease should have been enough.**
`triton/runtime/build.py:48` is `subprocess.check_call(cc_cmd,
stdout=subprocess.DEVNULL)` with no `stderr=`, and `CalledProcessError` carries
the argv and the return code and nothing else. Four attempts produced return
codes; none produced a compiler diagnostic. Wrapping that one call to print both
streams turned a three-day wall into one line of `gcc` output. **When a tool
reports a failure without the failing tool's own message, wrapping the call is
cheaper than any hypothesis you can form without it.**

**Rejected, and why.** `TORCH_DISABLE_NATIVE_JIT` exists — read out of the
installed `torch._native` rather than guessed — and would have removed the
Triton path without installing anything. It was not used: it moves the RoPE outer
product onto a different implementation, and a reference render whose kernels
were chosen to avoid provisioning a toolchain is not a reference. The line drawn
here is that a linker-search override (`TRITON_LIBCUDA_PATH`) is admissible and
an implementation switch is not.

**Why the render is driven by a committed script.** The four earlier attempts ran
an inline CLI inside a throwaway job. This one runs
`tools/oracle/ltx2_oracle.py`, so the file the row lands is the file that
produced the manifest, and the manifest's provenance is not a claim about a
script that never ran.

**What the defaults are.** Geometry 320x192x25 at 8 steps, seed 42, prompt fixed
in the script: the cheapest request that still drives the Gemma-4 tower, the
joint video+audio DiT, the conv video VAE and the audio VAE, held constant so
runs on different days compare. `--offload cpu` because `NONE` wants the whole
DiT resident and this repository has OOM-rebooted a GB10.

**What is NOT established.** One geometry, one prompt, one seed, one pipeline,
one offload mode, bf16 only. No comparison against this project's own render was
made; [#1854](https://github.com/mudler/vllm.cpp/issues/1854) owns that and now
has an absolute reference to ask for. The thirteen synthetic-weight golden
scripts are untouched, so every committed LTX-2.5 golden still comes from
upstream modules on PRNG weights.
