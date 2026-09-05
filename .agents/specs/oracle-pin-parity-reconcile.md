# ORACLE-PIN-PARITY-RECONCILE — one pin, two files, and an assertion that says which one is right

Issue: [#2829](https://github.com/mudler/vllm.cpp/issues/2829)
Row: `-` (no row owns it; `.agents/specs/upstream-pin-advance-e126687.md`
lists it under `## Owed`, and that entry stays there because it is the record
that the sync cycle owed this work)
Prior art: [#520](https://github.com/mudler/vllm.cpp/issues/520) and
[`bench-oracle-pin-reconcile.md`](bench-oracle-pin-reconcile.md), where a
duplicate of this same value drifted and the harness spent 17 days *refusing*
the oracle the record required; [#647](https://github.com/mudler/vllm.cpp/issues/647),
which built the oracle registry this change extends.

## Scope

**In scope.** One rule, added to `scripts/check-oracle-pins.py`: the vLLM
oracle record's `pin` and `pin_label` must agree with the ` ```parity-pin `
block in `.agents/upstream-sync.md`, which is the authority.

| surface | block | role | read by |
|---|---|---|---|
| `.agents/upstream-sync.md` | ` ```parity-pin ` | **authority** | `tools/bench/serve_low_common.py:read_parity_pin`, then `tools/bench/online_gate.py`, which refuses a mismatched oracle before any measurement |
| `.agents/oracles/vllm.md` | ` ```oracle-pin ` | copy | `scripts/check-oracle-pins.py` |

**Out of scope, deliberately.**

- **Deleting the copy.** That is the fix #2829 prefers and it is not available
  here. `pin` and `pin_label` are `REQUIRED_KEYS` of every oracle record, and
  `pin` is *load-bearing* in `check_record`: `gateable = yes` is refused when
  `pin == "UNPINNED"`. Removing the two keys from `vllm.md` alone would need a
  per-id exemption inside the shape rules that hold for the other thirteen
  oracles, and it would take the offline literal away from a registry whose
  whole point is that it can be read without a network. This change makes the
  copy *checked*, and #2829's better fix stays available on top of it.
- **Every other oracle.** `transformers`, `sglang`, `llama-cpp`, `ltx-2` and
  the rest hold their pin *only* in their own file. There is no `parity-pin`
  counterpart to reconcile them against, so applying this rule to them would
  fail all thirteen for being correct. AGENTS.md: a gate that fires on ordinary
  work is the defect, not the discipline. The reconciliation is scoped to
  `.agents/oracles/vllm.md` by filename stem.
- **Whether the pin exists upstream.** The checker stays network-free. This
  change compares two files in this tree and nothing else.
- **The prose surfaces.** Named below rather than covered.

## The third, fourth and fifth surfaces, named

The abbreviated revision and the release label also appear in prose, in three
public projections:

| file:line (at `c05d43299`) | text |
|---|---|
| `.agents/NOW.md:25` | ``Pin: vLLM `e126687a9a` (0.28.1rc1.dev132) since`` |
| `docs/FEATURES.md:14` | ``Reference versions: vLLM 0.28.1rc1.dev132 (`e126687a9a`, the parity pin since`` |
| `docs/benchmarks/how-we-measure.md:21` | ``**Oracle pin.** vLLM 0.28.1rc1.dev132 (`e126687a9a`) since 2026-09-03`` |

**They are not gated by this change, and the reason is not that they were
missed.** They carry no delimited block, they carry an *abbreviated* revision
whose length is not fixed, and they are sentences whose wording is free. A regex
over prose would have to be loose enough to find them and would then also match
the dozens of `.agents/sync/` and `.agents/specs/` files that name a *prior* pin
on purpose, which is a true historical fact this repository refuses to rewrite
(`bench-oracle-pin-reconcile.md` §"Out of scope"). Three of those files already
exist for `5559679229` and `e24d1b24`. So the gate would fire on correct work
on the day it landed.

They are listed under `## Owed` below and tracked by
[#2883](https://github.com/mudler/vllm.cpp/issues/2883), so that a later change
can decide to structure them rather than rediscover them. **The argument above
is narrower than it reads**, and #2883 says so: it rules out a regex over prose.
It does not rule out a rule that reads the authority and requires these three
*named paths* to contain the abbreviated revision and the public version, which
never looks at the other files and so cannot fire on a deliberate historical
mention. What such a rule still has to settle is the abbreviation length, which
nothing declares today.

## Design

`scripts/check-oracle-pins.py` gains one authority reader and one rule.

1. `parse_parity_pin(label, text, errors) -> dict[str, str] | None` parses the
   single ` ```parity-pin ` block out of the *text* of
   `.agents/upstream-sync.md`, using the same shape rules
   `tools/bench/serve_low_common.py` applies: exactly one block, `key = value`
   lines only, no unknown key, no duplicate key, no empty value. It **fails
   closed** — an absent, duplicated or unparsable block is an error, never a
   skipped rule. A rule that silently stops checking when it cannot find its
   expectation is a mute switch.

   **It takes text, not a path.** `main` opens the file and turns an `OSError`
   into the same kind of error, so the *unreadable* case is handled one level
   up. That keeps this function pure and testable against a synthetic authority
   with no temporary directory, which is what every case in
   `ParityReconciliationTests` relies on. `label` is the name to report under,
   supplied by the caller for the same reason. The name is
   `parse_parity_pin` and not `read_parity_pin` because
   `tools/bench/serve_low_common.py:read_parity_pin` already holds that name for
   the path-taking reader, and two functions with one name and different
   signatures is a trap for the next reader.
2. `check_parity_reconciliation(record, parity, errors)` compares the vLLM
   record against that dict:
   - `pin` must **equal** `vllm_commit`, exactly.
   - `pin_label` must **equal** the *public* part of `vllm_runtime_version`,
     that is the segment before the first `+`.
3. `main` calls both, selecting the record by filename stem `vllm.md`, and
   reports through the same error list as every other rule, so one red is one
   `oracle-pins FAILED` with rc 1.

**The expectation comes from the authority, never from the file under test.**
An anchor checker that reads its expectation out of the same file it checks is
a tautology and has shipped here before. `parity` is read from
`.agents/upstream-sync.md`; the value under test is read from
`.agents/oracles/vllm.md`; nothing crosses.

### The `pin_label` rule, and why it is equality and not a prefix

Today the two strings are `pin_label = 0.28.1rc1.dev132` and
`vllm_runtime_version = 0.28.1rc1.dev132+ge126687a9`. They are not equal, so a
rule is a choice and this spec states it.

**The rule is: `pin_label == vllm_runtime_version.partition("+")[0]`.** That is
exact equality after a *defined decomposition*, not a substring test. The `+…`
tail is the PEP 440 local version segment, which setuptools-scm appends as
`+g<sha>`; the part before it is the public version, which is what a release
label is. The distinction matters because a substring or `startswith` rule would
accept `pin_label = 0.28.1rc1.dev13` — a truncated, wrong label that is
nonetheless a prefix — and would accept `0.28` as well. The decomposition rule
refuses both, and it still accepts a future pin taken from a build with no local
segment at all, where `partition` returns the whole string and the rule degrades
to plain equality.

**Considered and declined:** also asserting that the `+g<sha>` segment prefixes
`vllm_commit`. It is true today (`e126687a9` prefixes
`e126687a9a828d…`) and it is `assert_oracle_commit`'s job against a *live*
runtime. As a static rule it would have to be conditional on the segment being
present, because a pin taken from a tagged release reports a bare `0.29.0` and
would go red while being correct. A conditional rule that is inert on the shape
it exists for is not worth the line, so this change does not add it. Named here
so the next reader knows it was a decision.

## Risks

| risk | disposition |
|---|---|
| The rule fires on the thirteen other oracles | Scoped by filename stem to `vllm.md`. Tested THROUGH `main`, which is where the scope lives: a synthetic registry holding the agreeing primary plus a secondary whose `pin` and `pin_label` name a revision and a release the authority never mentions returns rc 0. Replacing `main`'s scoped call with a loop over every registry file turns that case red |
| The rule silently stops checking when `upstream-sync.md` moves or its block is renamed | Fails closed. A missing record, a missing block, two blocks, or an unparsable line are each an error with its own message. Tested in both directions |
| The rule reads its expectation from the file it checks | Structurally impossible: `check_parity_reconciliation` takes the parity dict as a parameter and never opens a file |
| It blesses the duplication the checker's docstring warns against | Acknowledged in the docstring itself, which this change rewrites rather than leaves contradicting the code. The docstring now says the copy is checked *and* that removing it is still the better fix, and names #2829 |
| The `pin_label` rule accepts a wrong label | The decomposition rule refuses every proper prefix. Tested with `0.28.1rc1.dev13` |
| The self-test corpus no longer covers every rule | A second fixture corpus, `PARITY_FIXTURES`, is swept in both directions by `--self-test` alongside the existing one |

## Tests

`tests/scripts/test_check_oracle_pins.py` gains a `ParityReconciliationTests`
class. Every case is a mutation performed against a synthetic authority and a
synthetic record, so none of them depends on what the live pin happens to be:

1. the true pair reconciles with zero errors;
2. a `pin` that names the prior revision is reported;
3. a `pin_label` that names the prior release is reported;
4. a `pin_label` that is a proper prefix of the right one is reported — this is
   the case a substring rule would have accepted;
5. a `pin_label` equal to the *full* runtime string, local segment included, is
   reported;
6. a missing `parity-pin` block is reported;
7. two `parity-pin` blocks are reported;
8. an unparsable line, an unknown key, a duplicate key and an empty value are
   each reported;
9. an authority whose `vllm_runtime_version` carries no local segment at all
   still reconciles, which is the degenerate case the decomposition rule is
   chosen for;
10. the live tree reconciles, both through `main([])` and through the
    reconciliation alone over the two real files, with neither value written as
    a literal in the test;
11. **the rule is REACHED, and it is SCOPED.** `ParityThroughMainTests` points
    `main` at a synthetic registry — its own `ROOT`, `ORACLES`, `AGENTS_MD`,
    `UPSTREAM_SYNC` and `DECLARATION_ROOTS` — and asserts rc 0 on an agreeing
    pair, rc 1 on a disagreeing one, and rc 0 on an agreeing pair beside a
    secondary oracle pinned to something the authority never names. Every other
    case calls the rule directly, which proves the rule works and never that
    anything runs it. Deleting the `check_parity_reconciliation` call from
    `main` turns the second red, and only the second: the first asserts rc 0
    and gets rc 0 whether the rule runs or not. Widening the call into a loop
    over every registry file turns the third red. Both cases live here because
    both are properties of `main` and of nothing else: `check_registry` never
    invokes the parity rule, so a scoping test written against it cannot fail
    for the reason it names.

## Gates

```sh
python3 scripts/check-oracle-pins.py
python3 scripts/check-oracle-pins.py --self-test
python3 -m unittest tests.scripts.test_check_oracle_pins -v
python3 -m unittest tests.tools.test_oracle_pin
scripts/agent-preflight.sh
```

No build. This change touches one checker, one test module and this spec; it
compiles nothing and it runs on no device.

## Evidence

Red-before, green-after, and the restored tree, recorded in `## Outcome` when
the work lands.

## Stop conditions

- Stop and return `NEEDS_DECISION` if closing the duplication properly — the
  registry deriving `pin` from the authority at read time — turns out to be
  reachable without a per-id exemption. That is the fix #2829 prefers.
- Stop if the reconciliation cannot be made to go red by a mutation. A checker
  that cannot fail is a mute switch and must not be reported as a gate.
- Do not edit the pin values in `.agents/upstream-sync.md`. Another wave owns
  that file's content, and this change reads it and never writes it.

## Owed

- The three prose surfaces named in §"The third, fourth and fifth surfaces":
  `.agents/NOW.md`, `docs/FEATURES.md` and `docs/benchmarks/how-we-measure.md`
  restate the abbreviated pin and the release label, and nothing checks them
  either. Tracked by
  [#2883](https://github.com/mudler/vllm.cpp/issues/2883), which owns them once
  #2829 closes. That issue also records what the argument below does *not*
  establish: the 286-file measurement rules out a regex over prose, and not a
  rule over three named paths with a fixed extraction, which is what these
  surfaces are.
- The same reachability hole one call site over. Deleting
  `errors.extend(check_declarations(declarations, registry_ids))` from `main`
  leaves all 42 tests in `tests/scripts/test_check_oracle_pins.py` green at
  rc 0, so nothing proves `main` runs the declaration check either. It is
  pre-existing and not introduced by this row; `ParityThroughMainTests` is the
  pattern that closes it. Tracked by
  [#2898](https://github.com/mudler/vllm.cpp/issues/2898).
- The duplication itself. This change makes the copy checked; it does not
  remove it. #2829's preferred fix — the registry resolving `pin` for the
  primary oracle from the authority at read time — stays available and is
  strictly better than an assertion.

## Now

No row. This spec exists because a semantic checker change needs one, and it is
committed before the implementation.
