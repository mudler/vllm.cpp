# SPEC — pinning vLLM-Omni

**Row:** `ENG-UPSTREAM-OMNI-PIN`
**Issue:** [#633](https://github.com/mudler/vllm.cpp/issues/633)
**State:** `READY` — protocol change, no product code.

## Scope

Make the vLLM-Omni oracle *pinnable*, and state the two rules that follow from
the fact that it is a second pin rather than a second value of the vLLM one.

**This spec was rewritten after [#650](https://github.com/mudler/vllm.cpp/pull/650)
landed.** Its first draft proposed an `omni-parity-pin` block inside
[upstream-sync.md](../upstream-sync.md). #650 adopted the oracle registry —
one file per oracle under [`.agents/oracles/`](../oracles/), a fenced
`oracle-pin` block, and `scripts/check-oracle-pins.py` enforcing it — which is a
better answer to the same problem and explicitly rejects the shared-surface shape
the first draft had. So the registry is now the home of the pin, and this row no
longer proposes a record format at all.

In scope: the rules governing how the omni pin and the vLLM parity pin interact;
the answer to whether an omni pin advance re-opens the vLLM-side binding grids;
and the work owed to actually take the pin.

Out of scope: advancing the vLLM core pin to 0.27.0+; inventorying the omni model
surface; porting any omni architecture; re-stating anything AGENTS.md
§"When vLLM has no implementation" or [`.agents/oracles/README.md`](../oracles/README.md)
already says.

## Upstream chain

`vllm-project/vllm-omni` — a separate repository from `vllm-project/vllm`, on its
own release cadence, registry at `vllm_omni/model_executor/models/registry.py`.
Source-audited twice: `a4ea67a2` (v0.26.0) for MiniMax-H3
([minimax-h3.md](minimax-h3.md) §8.2), and
`bbe6ccc512a404a2df8c977ea29003002f2683e8`, which the Moss-TTS, Qwen3-TTS and
Higgs-Audio rows in [model-matrix.md](../model-matrix.md) anchor to (#609, #610).

Neither audit is a pin. Both read source, and a source read establishes what
exists, never what runs.

## Our baseline

[`.agents/oracles/vllm-omni.md`](../oracles/vllm-omni.md) exists and already
records the state honestly: `pin = UNPINNED`, `gateable = no`, `evidence = #633`.
So the *record* is no longer missing. What is still missing is the pin itself,
and with it any oracle-run gate for MiniMax-H3 W3+, LTX-2.5 and the whole TTS
family including IndexTTS-2.5 ([#634](https://github.com/mudler/vllm.cpp/issues/634)).

vLLM-Omni requires vLLM **0.27.0+**; our parity pin is `555967922` (0.26.0.dev0).
That conflict is the reason this cannot be handled by advancing one number.

## Port map

Nothing is ported. Two record changes, and one deliberate non-change:

| Change | Where |
|---|---|
| The omni pin's home is the oracle registry, not this file; do not add a second pin block here | [upstream-sync.md](../upstream-sync.md) |
| The two omni rules (below) | [upstream-sync.md](../upstream-sync.md) §Rules |
| **No edit** to `.agents/oracles/vllm-omni.md` | it already says `UNPINNED` / `gateable = no` / `evidence = #633`, which is true |

**Rule 1 — the two pins may legitimately disagree, and an omni-gated number is
labeled with both.** vLLM-Omni needs 0.27.0+ while the parity pin is 0.26.0.dev0.
Forcing them equal would mean advancing the core pin — touching every gated row
and every binding number — to satisfy a lane that touches none of them.

**Rule 2 — an omni-gated result is never evidence about the core pin's surface.**
Not in a vLLM-side parity claim, not in a binding grid, not in a
`docs/BENCHMARKS.md` row owned by a core-pinned row. Rule 1 is what makes Rule 2
necessary: once the two pins can differ, a number carried across is a number
measured against a different dependency tree.

Per-architecture gateability is NOT restated here: AGENTS.md already requires an
oracle to demonstrably build and run the model, and
[`.agents/oracles/README.md`](../oracles/README.md) already binds `gateable = yes`
to that. The only omni-specific consequence worth recording is that one
`gateable = yes` does not generalize across ~40 architectures, which is why the
demonstration is recorded on the owning row rather than the oracle file.

## Tests to port

None from upstream — local protocol machinery, no vLLM analogue, so the mirror
rule does not apply.

| Assertion | Proves | Owed by |
|---|---|---|
| the `ENG-UPSTREAM-OMNI-PIN` row and the `ENGINE_ROWS` bump arrive together | a count moved for a row, not to silence a failure | this PR (landed) |
| an omni-gated evidence row cited by a core-pinned row goes red | Rule 2 is enforced by a checker, not by prose | W3 |
| a `core_commit_used` value that does not match the `vllm.__version__` in the same interpreter goes red | the isolation condition under Gates actually fires | W4 |

`check-oracle-pins.py` already covers the block's own shape, and this row does
not duplicate it.

## Gates

**Does advancing the omni pin re-open the vLLM-side binding grids? No —
conditional on isolation.** The conditions:

- the omni oracle lives in its own virtualenv, never the venv the parity pin's
  oracle runs from;
- no core-pinned row's evidence cites an omni-gated number (Rule 2);
- the omni install mutates neither the shared checkout, `${VLLM_SOURCE}`, nor the
  `vllm-oracle-next` environment the parity pin measures itself from.

Fail any condition and the environments are entangled: what moved is the
denominator's own dependency tree, so every binding grid measured afterwards is
suspect and the advance is re-validated as a core sync cycle.

The isolation is asserted, not assumed. The identity check reads
`vllm_omni.__version__` **and** the `vllm.__version__` visible in the same
interpreter. Recording the latter needs a key the `oracle-pin` schema does not
have yet — deliberately not invented here, because the pin is `UNPINNED` and
there is no value to record. Adding `core_commit_used` to the schema, and
teaching `check-oracle-pins.py` about it with its own mutation evidence, is part
of W4 and lands with the measurement.

## Dependencies

An eligible host, and nothing else in this repository. The pin values are
**PENDING**: they require installing vLLM-Omni somewhere and measuring
`vllm_omni.__version__` there, and an unavailable value stays `PENDING` rather
than becoming an assumption.

Exact handoff: create a dedicated venv (never `vllm-oracle-next`), install
vllm-omni at the chosen commit with the vLLM version it requires, record `pin`,
`pin_label`, `pinned_on` and the new `core_commit_used` from the running
interpreter, and prove gateability for at least one architecture by generating
output from it.

Note for that operator, current as of 2026-08-13: dgx `/home` was reported at 99%
with two reboots that morning under multi-session GPU load. ENOSPC on that host
has previously produced **false policy refusals citing a retired rule**, so a red
result there is evidence about the disk until it is back under 95%.

## Work breakdown

| W | Work | Depends on |
|---|---|---|
| W1 | Point `upstream-sync.md` at the registry; land the two omni rules | — |
| W2 | ~~Scope the exhaustive-inventory claim~~ — **superseded**, #609/#610/#650 did it | — |
| W3 | Checker: an omni-gated number cited by a core-pinned row goes red | W1 |
| W4 | Measure the pin; extend the `oracle-pin` schema with `core_commit_used` and teach the checker, with mutation evidence | W1, a host |
| W5 | Prove gateability for one architecture by generating output from it | W4 |

W1 is this PR. W3-W5 are owed and unclaimed.

## Risks/decisions

- **Scope creep into a core pin advance.** vLLM-Omni wanting 0.27.0+ is a
  standing pull toward moving the parity pin, which touches every gated row and
  every binding number. **Stop** if the work cannot proceed without moving it,
  and reconcile that as its own row.
- **Two pins read as one.** The failure Rule 2 exists to prevent. W3 gives it a
  test; until then it is prose, and prose drifts.
- **A pin that cannot run anything.** Pinning a commit whose models do not run on
  our hardware records a revision and unblocks nothing. `gateable` is what keeps
  the pin from being mistaken for capability.
- **This spec was already superseded once, mid-flight.** #650 landed the registry
  while this row was open, and the first draft's central proposal became the
  wrong shape. Re-verify against `main` before implementing W3-W5; that is the
  rule this row has already been caught by once.

## Now

`READY`, unclaimed. W1 lands with this spec; W2 is superseded; W3-W5 are not
started and the pin values are PENDING an eligible host.

Downstream: [#634](https://github.com/mudler/vllm.cpp/issues/634) (IndexTTS-2.5)
is planned behind this row; MiniMax-H3 W3+ and LTX-2.5 carry the same blocker and
unblock together.
