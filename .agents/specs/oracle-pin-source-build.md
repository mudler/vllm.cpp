# ORACLE-PIN-SOURCE-BUILD — the pin's two version strings are equal now, and the test that said they cannot be

Issue: [#2931](https://github.com/mudler/vllm.cpp/issues/2931)
Row: `ORACLE-PIN-SOURCE-BUILD`
Prior art: [#520](https://github.com/mudler/vllm.cpp/issues/520) and
[`bench-oracle-pin-reconcile.md`](bench-oracle-pin-reconcile.md), which
introduced the assertion this row replaces;
[#2896](https://github.com/mudler/vllm.cpp/issues/2896) and `5d97007c2`, the
developer-instructed sync that made it red;
[`oracle-pin-parity-reconcile.md`](oracle-pin-parity-reconcile.md), the sibling
row over the same record.

## The failure

`main` is red at `c796fea41` on `tools suites`, which
`scripts/agent-preflight.sh:561` runs in every preflight, so every row that
merges current `main` inherits it:

```
FAIL: test_metadata_and_runtime_strings_differ_on_the_pin
      (tests.tools.test_oracle_pin.ParityPinRecordTests)
  File "tests/tools/test_oracle_pin.py", line 86
    self.assertNotEqual(VLLM_DISTRIBUTION_VERSION, VLLM_ORACLE_VERSION)
  AssertionError: '0.28.1rc1.dev132+ge126687a9' == '0.28.1rc1.dev132+ge126687a9'
```

## The ruling: the invariant is stale, the landed sync is correct

#2931 named two possible repairs and said only the owning row can choose. It is
the second one, and each link was verified against the tree rather than
inherited:

1. **The sync is a developer instruction.** `5d97007c2` records it verbatim:
   *"fix the version string to the source-build value"*, 2026-09-04, `Fixes: #2896`.
2. **Its value is measured, not transcribed.**
   `.agents/sync/2026-09-03-e126687-runhalf.md` line 52 is
   `BUILD_ENV VLLM_USE_PRECOMPILED=0 VLLM_TARGET_DEVICE=cuda` and line 56 is
   `WHEEL=…/vllm-0.28.1rc1.dev132+ge126687a9-cp312-cp312-linux_aarch64.whl`.
   A source build at this revision carries **no** local suffix, so its
   distribution version and its runtime version are the same string.
3. **The old value made the harness unsatisfiable.**
   `tools/bench/online_gate.py:3531-3534` compares `metadata_version` against
   `VLLM_DISTRIBUTION_VERSION` for **equality**. On aarch64 the only build mode
   that produced `…​.precompiled` leaves an install with no compiled extensions,
   so the only string the gate accepted named a build that cannot execute a
   kernel.
4. **What the assertion was built to catch is a DIFFERENT thing than what it
   asserted.** `356fa7750` (#520) states it in its own body, hole 2:

   > Distribution metadata is matched against its OWN recorded string. The two
   > differ on the pin (metadata appends ".precompiled"), so the previous
   > `metadata == runtime == CONST` shape was unsatisfiable there at ANY value,
   > and a `startswith` would have accepted an arbitrary suffix instead.

   `git show 356fa7750^:tools/bench/online_gate.py` line 3509-3510 is that
   previous shape, exactly:

   ```python
   metadata_version != VLLM_ORACLE_VERSION
   or runtime_version != VLLM_ORACLE_VERSION
   ```

   **One constant served both fields.** That is the regression the test exists
   to catch. Strict inequality of the two recorded strings was the *symptom*
   that made the one-constant shape unsatisfiable at the pin of the day; it was
   never the guarantee. A source build legitimately erases the symptom and
   leaves the guarantee untouched.

So the record is right and the test's premise is stale. This row replaces the
premise without weakening the guarantee.

## Scope

**In scope.**

- `tests/tools/test_oracle_pin.py`: replace
  `test_metadata_and_runtime_strings_differ_on_the_pin` with (a) a record-level
  test that keeps the durable prefix relation, and (b) a **behavioural** test in
  `OracleIdentityIsWiredIntoEveryEntryPointTests` that fails against the
  `metadata == runtime == CONST` shape itself.
- Five code comments the landed sync falsified, corrected in the same change so
  no reader is told a fact the tree contradicts: `tools/bench/online_gate.py:3527-3530`,
  `tools/bench/serve_low_common.py:42-46` and `:104-106`, the inline comment at
  `tests/tools/test_oracle_pin.py:153`, and
  `tools/bench/gdn_packed_component.py:1596-1598`, which sits above the same
  two-field comparison and carried the same present-tense falsehood.

**Out of scope, deliberately.**

- **Editing the pin.** `.agents/upstream-sync.md` is not written by this row.
  The value is a developer instruction backed by a recorded measurement; if it
  were wrong that would be a `NEEDS_DECISION`, not an edit. This row reads it.
- **Deleting the assertion.** AGENTS.md forbids making a red gate green by
  removing a check. The replacement must fail against the condition the original
  guarded, and §"Tests" shows it doing so.
- **Building or running vLLM.** The measurement this row depends on already
  happened and is recorded in the tree.
- **`scripts/check-oracle-pins.py`.** Its `.precompiled` strings are synthetic
  fixtures (`:406`, `tests/scripts/test_check_oracle_pins.py:82,373`) and assert
  nothing about the live pin. They stay.

## Upstream anchors

None. This is a harness-identity rule over this repository's own oracle pin
record. vLLM defines the version strings it emits; it does not define how this
tree records or compares them.

## Design

### What the guarantee actually is

The record carries the distribution and runtime versions as **two separately
sourced fields**, and each entry point compares each observed string against
**its own** recorded field. Three properties follow, and each needs a home:

| property | where it is held | status |
|---|---|---|
| both fields exist in the record, as distinct keys | `_PIN_FIELDS` plus `read_parity_pin`'s "omits" refusal, exercised by `test_missing_field_is_fatal` and `test_constants_are_the_pin_record_not_a_copy` | already held, unchanged |
| the distribution string extends the runtime string | `test_the_distribution_string_extends_the_runtime_string` | kept, docstring rewritten |
| **each observed string is compared against its own constant** | *nothing, today* | **the hole this row closes** |

The third row is the one that matters, and it was never actually tested. The
deleted `assertNotEqual` asserted over *data* and so could only ever prove that
the two recorded strings happened to differ. It said nothing about whether
`record_oracle_manifest` consults both.

### Why no existing test covers it, and why the new one must patch a constant

At today's pin `VLLM_DISTRIBUTION_VERSION == VLLM_ORACLE_VERSION`. Enumerate the
two existing wired-in cases against the defective one-constant shape
(`CONST = VLLM_ORACLE_VERSION`):

| case | metadata | runtime | correct code | one-constant code |
|---|---|---|---|---|
| `test_record_oracle_refuses_a_rollback_runtime_version` | pin | rollback | raises | raises |
| `test_record_oracle_refuses_rollback_distribution_metadata` | rollback | pin | raises | raises |

Neither discriminates. **While the two recorded strings are equal, no real input
can separate the two shapes**, because the correct code's two comparisons are
against the same value. This is precisely the situation
`OracleIdentityIsWiredIntoEveryEntryPointTests` already documents for the commit
term — *"at today's pin the exact equality strictly dominates the commit
assertion … so no input can reach the assertion while the equality holds.
Patching the constants models the pin shape where it is the operative term"* —
and the new case follows that established pattern rather than inventing one.

`test_record_oracle_compares_each_string_against_its_own_constant` patches
`tools.bench.online_gate.VLLM_DISTRIBUTION_VERSION` to
`VLLM_ORACLE_VERSION + ".precompiled"` — the shape that was *real* at the
previous pin, not an invented one — and then feeds
`metadata = runtime = VLLM_ORACLE_VERSION`. Now:

- correct code: `metadata != VLLM_DISTRIBUTION_VERSION` → raises
  `oracle version drift`; and
- one-constant code: both equal `CONST` → **accepts**, and the assertion is red.

A second case pins the polarity in the other direction: with the same patch,
`metadata = ORACLE + ".precompiled"`, `runtime = ORACLE + ".precompiled"` must
raise, because the *runtime* term must still be compared against
`VLLM_ORACLE_VERSION`. That case is red if the runtime comparison is deleted or
re-pointed at the distribution constant.

Both cases match the exact refusal string, not bare `HarnessError`, for the
reason the enclosing class already gives: this function raises `HarnessError` for
a dozen unrelated reasons, and an unanchored `assertRaises` would stay green on a
gutted check that merely failed later for want of a fixture.

### The prefix relation survives, and is the durable half

`VLLM_DISTRIBUTION_VERSION.startswith(VLLM_ORACLE_VERSION)` holds at a source
build (identically) and at a precompiled build (`+ ".precompiled"`). It is the
relation that is true of the *design* rather than of one build mode, so it stays,
under a name that says what it means.

## Risks

| risk | disposition |
|---|---|
| The replacement is a mute switch — it cannot fail | Shown failing in §"Tests" red-before against the exact pre-#520 `metadata == runtime == CONST` diff, restored byte-for-byte after |
| The replacement is narrower than what it replaced | It is strictly wider: the old assertion could not observe `online_gate.py` at all, and the new one executes the call site. The record-level half it drops (strict inequality) is a property of one build mode, and §"The ruling" shows why it is not the guarantee |
| The `.precompiled` value in the new test becomes a fourth copy of the pin | It is not a pin value. It is `VLLM_ORACLE_VERSION + ".precompiled"`, derived from the constant at test time, so it tracks any future pin advance and no literal version string is written |
| Correcting the five comments hides a real intent | Each is corrected to state what is now true *and why it changed*, naming the source build, not simply deleted |
| A future precompiled-mode pin re-breaks this | It cannot. Both new cases are patched, so they are independent of what the live pin holds — which is exactly the defect in the assertion being replaced |
| The distribution field is quietly dropped from the record because "it equals the other one" | `read_parity_pin`'s `omits` refusal plus `test_missing_field_is_fatal`, unchanged. Named here so the next reader knows it is held elsewhere and not by this row |

## Tests

`tests/tools/test_oracle_pin.py`, all in `tests/tools`, no device and no build:

1. `ParityPinRecordTests.test_the_distribution_string_extends_the_runtime_string`
   — the surviving prefix relation, with a docstring recording that strict
   inequality was an artefact of `VLLM_USE_PRECOMPILED=1` and naming `5d97007c2`.
2. `OracleIdentityIsWiredIntoEveryEntryPointTests.test_record_oracle_compares_each_string_against_its_own_constant`
   — the discriminating behavioural case described in §"Design".
3. `…test_record_oracle_compares_the_runtime_string_against_the_runtime_constant`
   — the opposite polarity.

**Red-before (the mutation that proves case 2 is not a mute switch).** In a
scratch copy, restore the pre-#520 one-constant shape at
`tools/bench/online_gate.py:3531-3534`:

```python
    if (
        metadata_version != VLLM_ORACLE_VERSION
        or runtime_version != VLLM_ORACLE_VERSION
    ):
```

Case 2 must go **red** and the two pre-existing rollback cases must stay green,
which is the demonstration that they never covered this. Restore the tree
byte-for-byte with `git checkout --` and re-verify green.

**Second mutation.** Delete the `runtime_version != VLLM_ORACLE_VERSION` term.
Case 3 must go red.

## Gates

```sh
python3 -m unittest tests.tools.test_oracle_pin -v
python3 -m unittest discover -s tests/tools -t . -p "test_*.py"
python3 scripts/check-oracle-pins.py
python3 scripts/check-oracle-pins.py --self-test
python3 scripts/agent-issue-index.py --refresh && python3 scripts/check-agent-record.py
python3 scripts/check-commit-style.py --range origin/main..HEAD
python3 scripts/check-pr-size.py --base origin/main --head HEAD
scripts/agent-preflight.sh
```

No compile. This row edits one test module and five comments.

## Evidence

Recorded in `## Outcome` when the row lands: the literal `tools suites` line
from the final preflight, the red-before output of each mutation with the
restored-tree diff shown empty, and the green-after.

## Stop conditions

- **Return `NEEDS_DECISION`, do not edit `.agents/upstream-sync.md`,** if the
  recorded build evidence turns out not to support the landed value. It does;
  §"The ruling" links 2 and 3 are the check.
- **Stop if the replacement cannot be made red** by the pre-#520 mutation. A
  check that cannot fail is not a replacement, and reporting it as one would be
  the failure this row exists to prevent.
- Do not build or run vLLM. Do not take a GPU lease.

## Owed

- **The `.precompiled` build mode is still not proven unrunnable by anything
  executable.** `.agents/upstream-sync.md` now asserts in prose that *"a build
  that carries `.precompiled` on aarch64 is not a runnable oracle"*, resting on
  a 13,872-byte editable install observed once. Nothing in `tests/` holds that,
  and nothing can without a build. It is the reason the pin has the value it
  has, so it deserves better than prose. **This bullet is the owner.**
  [#2931](https://github.com/mudler/vllm.cpp/issues/2931) raised it and this row
  closes that issue, so after the merge nothing but this entry carries the debt;
  it is written to stand without it. The next person to build the oracle files
  the follow-up issue and points it back here, and
  `.agents/upstream-sync.md` already instructs that builder to record which
  build mode they used.
- **`tools/bench/serve_low_common.py:135` names a pin sha the record advanced
  past.** The `assert_oracle_commit` docstring says *"today that constant
  already CONTAINS `+g555967922`"*; `VLLM_ORACLE_VERSION` has read
  `0.28.1rc1.dev132+ge126687a9` since `e8467758e` (2026-09-03) moved the parity
  pin. It was falsified by that **pin advance**, not by the `5d97007c2` sync
  this row answers, so it is outside the row's scope and the row does not touch
  it. Filed as [#2949](https://github.com/mudler/vllm.cpp/issues/2949), which
  names no owning row and needs one; it is listed here so it is not orphaned.
- **The token gate has still not run at this pin.** `5d97007c2` says so and this
  row changes nothing about it: every committed golden predates `e126687a9a`.
  Owned by `.agents/specs/upstream-sync-headpin-tokengate.md`, not by this row.

## Now

`ORACLE-PIN-SOURCE-BUILD` is `ACTIVE` on the repair of the stale invariant, and
`DONE` when `tools suites` is green on a tree that has merged `origin/main`. The
row has no roadmap entry and needs none: it owns one test module's premise and
the five comments the sync falsified, and its lifecycle is this file.
