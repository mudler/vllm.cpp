# SPEC — a parity pin for vLLM-Omni

**Row:** `ENG-UPSTREAM-OMNI-PIN`
**Issue:** [#633](https://github.com/mudler/vllm.cpp/issues/633)
**State:** `READY` — protocol change, no product code.

## Scope

Extend [upstream-sync.md](../upstream-sync.md) so a second upstream repository
can be pinned, and so a row whose architecture exists only in vLLM-Omni has a
protocol-legal oracle to be gated against.

In scope: the pin record and its identity assertion; how the two pins interact;
the gateability rule; the inventory-scope correction in
[model-matrix.md](../model-matrix.md); and an explicit answer to whether an omni
pin advance re-opens the vLLM-side binding grids.

Out of scope: advancing the vLLM core pin to 0.27.0+; inventorying the omni model
surface; porting any omni architecture. Risks/decisions records why the first of
those must stay separable.

## Upstream chain

`vllm-project/vllm-omni` — a separate repository from `vllm-project/vllm`, on its
own release cadence, with its own registry at
`vllm_omni/model_executor/models/registry.py`. This project has source-audited it
twice: at `a4ea67a2` (v0.26.0) for MiniMax-H3 ([minimax-h3.md](minimax-h3.md)
§8.2), and more recently at `bbe6ccc512a404a2df8c977ea29003002f2683e8`, which the
Moss-TTS, Qwen3-TTS and Higgs-Audio rows in [model-matrix.md](../model-matrix.md)
anchor to (#609, #610). Neither audit is a pin: both read source, and a source
read establishes what exists, never what runs.

Roughly 40 modules there have no counterpart in the vLLM registry our inventory
was derived from: the whole speech-synthesis family (IndexTTS2, Fish Speech,
GLM-TTS, Ming-TTS, MOSS-TTS, Qwen3-TTS, VoxCPM2, Voxtral-TTS, Higgs Audio v2/v3,
Step Audio2, CosyVoice3, OmniVoice) plus Qwen2.5/3-Omni, Aura-Omni,
Ming-Flash-Omni and Hunyuan Image3/Video.

## Our baseline

[upstream-sync.md](../upstream-sync.md) pins exactly one repository. Every
architecture living only in vLLM-Omni therefore has no nameable oracle — not a
hard comparison, an unavailable one. Three places already record this as the
blocker:

- [minimax-h3.md](minimax-h3.md):205-208 — "a vllm-omni pin is a prerequisite for
  W3+".
- `MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit` in
  [model-matrix.md](../model-matrix.md) closes with "OPEN: there is no vllm-omni
  parity PIN — the upstream-sync protocol covers only the vLLM repo".
- [#435](https://github.com/mudler/vllm.cpp/issues/435) (LTX-2.5) needs the same.

It is also a records defect, though a shrinking one. `model-matrix.md` claims an
**exhaustive** architecture inventory, and that claim is true of one repository
without saying so. Since this row was opened, #609/#610 rowed several omni
architectures explicitly as out-of-repo, so the surface is no longer wholly
uninventoried — what remains is that the document's own scope sentence does not
state the boundary those rows are working around, and every one of them records
the same missing-pin blocker in its evidence cell. Fixing the sentence is this
row's W2; inventorying the rest is not this row's job.

## Port map

Nothing is ported; this row changes protocol machinery. The three record edits:

| Change | Where |
|---|---|
| `omni-parity-pin` block: `vllm_omni_commit`, `vllm_omni_runtime_version`, `vllm_omni_requires_vllm`, `vllm_core_commit_used` | [upstream-sync.md](../upstream-sync.md) |
| Omni-pin concept, the two-pin rules, and the isolation rule | [upstream-sync.md](../upstream-sync.md) Concepts + Rules |
| Exhaustive-inventory claim scoped to the vLLM repo; omni surface recorded as a separate obligation, excluded from totals | [model-matrix.md](../model-matrix.md) |

**Identity.** The core block's permanent constraint carries over verbatim: the
runtime version must contain a `+g<sha>` segment prefixing `vllm_omni_commit`,
and the value is taken from a **measured** `vllm_omni.__version__` on the oracle
host, never transcribed from a release number. The failure this prevents is on
record — a duplicated, transcribed pin drifted once and the harness spent 17 days
refusing the oracle the record required (#520).

**Two pins, allowed to disagree.** vLLM-Omni requires vLLM **0.27.0+**; our core
pin is `555967922` (0.26.0.dev0). Forcing one pin would mean advancing the core
pin to satisfy a lane that touches none of the gated rows. So the pins are
independent and `vllm_core_commit_used` records what the omni oracle actually ran
against. Two rules follow, and they are the point of recording it:

1. A result gated against the omni oracle is labeled with **both** commits.
2. An omni-gated result is **never** evidence about the core pin's surface — not
   in a vLLM-side parity claim, a binding grid, or a `docs/BENCHMARKS.md` row
   owned by a core-pinned row.

**Gateability is per ARCHITECTURE, not per pin.** An omni pin becomes gateable
for a given model only once the oracle demonstrably builds, runs and emits output
for it. Constructing a config, resolving a checkpoint or importing the module
proves nothing; `assert_oracle_commit` exists because an oracle once resolved and
was a rollback. Pinning the repo does not make 40 architectures gateable — it
makes each gateable as its own demonstration lands, recorded on the owning row.

## Tests to port

None from upstream — this is local protocol machinery with no vLLM analogue, so
the mirror rule does not apply. The checker assertions owed, each RED-first:

| Assertion | Proves |
|---|---|
| the `omni-parity-pin` block parses and every field is present | a partially-filled block is refused, not silently half-read |
| a `vllm_omni_runtime_version` whose `+g<sha>` does not prefix `vllm_omni_commit` goes red | the core block's assertion carries over; mutate the sha and it must fail |
| an identity check where `vllm.__version__` != `vllm_core_commit_used` goes red | the isolation assertion under Gates actually fires |
| an omni-gated evidence row cited by a core-pinned row goes red | rule 2 above is enforced by a checker, not by prose |
| `model-matrix.md` totals unchanged by this row | the inventory correction is a scope statement, not a renumbering |

Each is mutated in a scratch copy and proven to fail for its own reason before
the change lands. A red gate is never turned green by deleting an assertion or
widening a scope.

## Gates

**Does advancing the omni pin require re-running the vLLM-side binding grids?
No — conditional on isolation.** The conditions:

- the omni oracle lives in its own virtualenv, never the venv the core pin's
  oracle runs from;
- no core-pinned row's evidence cites an omni-gated number;
- the omni install mutates neither the shared checkout, `${VLLM_SOURCE}`, nor the
  `vllm-oracle-next` environment the core pin measures itself from.

If any condition fails the environments are entangled and every binding grid
measured afterwards is suspect, because what moved is the denominator's own
dependency tree. The advance is then treated as a core sync cycle and
re-validated as one.

The isolation is asserted, not assumed: the identity check reads
`vllm_omni.__version__` **and** the `vllm.__version__` visible in the same
interpreter, and refuses if the latter is not `vllm_core_commit_used`.

## Dependencies

An eligible host, and nothing else in this repository. The concrete pin values
are **PENDING** and this spec does not invent them: they require installing
vLLM-Omni somewhere and measuring `vllm_omni.__version__` there. An unavailable
value stays `PENDING` rather than becoming an assumption.

Exact handoff for whoever measures it: create a dedicated venv (never
`vllm-oracle-next`), install vllm-omni at the chosen commit with the vLLM version
it requires, record all four fields from the running interpreter, and prove
gateability for at least one architecture by generating output from it.

Note for that operator, current as of 2026-08-13: dgx `/home` was reported at 99%
with two reboots that morning under multi-session GPU load. ENOSPC on that host
has previously produced **false policy refusals citing a retired rule**, so a red
result there is evidence about the disk until it is back under 95%.

## Work breakdown

| W | Work | Depends on |
|---|---|---|
| W1 | `upstream-sync.md`: the block schema, the concept, the two-pin rules and the isolation rule | — |
| W2 | `model-matrix.md`: scope the exhaustive claim to the vLLM repo; record the omni surface as a separate obligation excluded from totals | — |
| W3 | Checker + the five RED-first assertions under Tests to port | W1 |
| W4 | Measure the pin on an eligible host and fill the block | W1, a host |
| W5 | Prove gateability for one architecture by generating output from the oracle | W4 |

W1 and W2 are this PR's scope. W3-W5 are owed and unclaimed.

## Risks/decisions

- **Scope creep into a core pin advance.** vLLM-Omni wanting 0.27.0+ is a
  standing pull toward moving the core pin, which touches every gated row and
  every binding number. **Stop** if the work cannot proceed without moving it,
  and reconcile that as its own row.
- **Two pins read as one.** The failure this invites is quoting an omni-gated
  number in a core-pinned claim. The rules above make it a rule and Tests to port
  gives it a test; without the test it is prose, and prose drifts.
- **A pin that cannot run anything.** Pinning a commit whose models do not run on
  our hardware records a number and unblocks nothing. Per-architecture
  gateability is what stops the pin being mistaken for capability.
- **Inventory debt made official.** Recording the omni surface as a known
  obligation must not decay into permanent report-only state. It is visible debt.

## Now

`READY`, unclaimed. W1-W2 land with this spec; W3-W5 are not started, and the pin
values themselves are PENDING an eligible host.

Downstream: [#634](https://github.com/mudler/vllm.cpp/issues/634) (IndexTTS-2.5)
is planned behind this row; MiniMax-H3 W3+ and
[#435](https://github.com/mudler/vllm.cpp/issues/435) (LTX-2.5) are blocked on the
same prerequisite and unblock together.
