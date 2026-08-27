# Spec — two gates around the SGLang lease identity suite that measure nothing

Issue: [#1832](https://github.com/mudler/vllm.cpp/issues/1832),
[#1833](https://github.com/mudler/vllm.cpp/issues/1833)
Row: `GATE-SGLANG-MANIFEST-AND-SUITE-REGISTRATION` (unplaced gate defect; both
findings are properties of checkers, not of the `SGLANG-ORACLE-LEASE-WHEEL`
capability, which is `DONE`)
State: `ACTIVE`

Both defects were raised by the fresh review of PR #1831 (`2a9de2eae`) and were
deliberately not repaired there. They are carried under `## Owed` in
[`sglang-wheel-in-lease.md`](sglang-wheel-in-lease.md).

## Scope

### #1832 — the manifest count is compared to itself

`.agents/specs/sglang-wheel-in-lease.json` is the only identity assertion the
SGLang oracle has, because `sglang/_version.py` in the published wheel sets
`__commit_id__ = None`. The one test that touches its population is
`tests/scripts/test_sglang_lease_identity.py::test_file_count_agrees_with_the_file_table`:

```python
self.assertEqual(self.manifest["file_count"], len(self.manifest["files"]))
```

That compares one JSON document to itself. It cannot see a manifest that
asserts a *different tree*, and it goes green on the exact shape a
mis-generated manifest takes. Measured on this branch's base
(`331eda8887e6a5c06244944c328b949b035cce4a`), each mutation proven applied by
`git diff --numstat` and each restored by sha256:

| mutation | `file_count` / `len(files)` | suite |
|---|---|---|
| `files` emptied, `file_count` set to `0` | 0 / 0 | `Ran 14 tests ... OK`, **rc=0** |
| `sglang/README.md` dropped, `file_count` set to `3337` | 3337 / 3337 | `Ran 14 tests ... OK`, **rc=0** |

`3338` is quoted as measured in three records — `.agents/environment.md:284`,
`.agents/oracles/sglang.md:148` and the `SGLANG-ORACLE-LEASE-WHEEL` row of
`.agents/sglang-matrix.md` — while `grep -rn 3338 scripts/ tests/scripts/
.github/` exits 1. A number that no executing code holds is a number that
drifts, and this one is the denominator of `IDENTITY_RC=0`.

In scope: the count becomes an **assertion**, held as a literal in the checking
file and never read back out of the file it checks, and the three records are
bound to that same literal so they cannot drift from it either.

### #1833 — both registrations of the suite are deletable at rc=0

`tests/scripts/test_sglang_lease_identity.py` is registered twice, deliberately:
in the `SUITES` array of `scripts/agent-preflight.sh` and as its own step in the
`agent-record` job of `.github/workflows/ci.yml`. Neither registration is held
by anything. Measured on the same base:

| mutation | proof it applied | `scripts/check-test-registration.py` |
|---|---|---|
| delete `  test_sglang_lease_identity` from `SUITES` | `0 1` numstat | `OK: ...`, **rc=0** |
| delete the whole 11-line CI step | `0 11` numstat, `yaml.safe_load` still parses, `grep -c` returns `0` | `OK: ...`, **rc=0** |

In scope: `wiring_errors` in `scripts/check-test-registration.py` gains a pinned
set of Python suites that must appear in **both** lanes, seeded with this one,
plus its red-first mutation cases.

## Out of scope, and why

**The re-derivation of the manifest.** #1832's honest repair is a second,
independent install inside an `rc` lease on `dgx:gpu0` that regenerates the
manifest and diffs it against the committed one. No checker reading the
committed JSON can do that, and this row takes no GPU lease: another session is
measuring SGLang c=8 on that device right now. What this row changes is the
*strength of the gate*, not the *status of the oracle*. `gateable = yes` is
untouched, `#1265` stays open, and the owed re-derivation stays owed — the
`## Owed` bullet is narrowed to say exactly what is left rather than deleted.

**The population rule for every Python suite.** #1833 suggests "every
`tests/scripts/test_*.py` must appear in `SUITES` and in a CI lane". That rule
is correct and it is **not** taken here, because it is a repo-wide sweep with a
measured cost. Read on this branch's base with the checker's own
`_bash_array_values` and `_active_ci_commands`:

- `SUITES` holds 64 entries; CI's unconditional run blocks execute 65 suites
  under `tests/scripts/`.
- 11 are in preflight only: `test_ab_arms_differ`, `test_agent_gates`,
  `test_agent_onboard`, `test_agent_pr_body`, `test_agent_role`,
  `test_agent_start`, `test_audit_live_rows`, `test_check_prompt_contract`,
  `test_gpu_lock_one_truth`, `test_ltx2_oracle_goldens`,
  `test_rc_stage_checkpoint`.
- 12 are in CI only: `test_check_gemv_invocation_consistency`,
  `test_check_now_current`, `test_check_site`, `test_ci_site_lane`,
  `test_convert_glm5_next_gguf`, `test_indextts2_config_contract`,
  `test_indextts2_convert`, `test_indextts2_emotion_arch_covered`,
  `test_indextts2_pth_manifest`, `test_indextts2_reference_path`,
  `test_ltx25_render_compare`, `test_vocoder1d_single_home`.

A symmetric population rule therefore reds on 23 suites at once, and each of
those 23 is a separate decision: some belong in one lane on purpose (an
`rc`-staging suite has no business in CI; a site-lane suite has no business in a
local preflight), and some are the genuine gap
[#408](https://github.com/mudler/vllm.cpp/issues/408) and
[#1509](https://github.com/mudler/vllm.cpp/issues/1509) already own. Landing the
population rule inside this row would either force 23 unreviewed
classifications into one branch or force a floor low enough to be a mute
switch, which is the `NORETURN_POPULATION_FLOOR = 40`-against-53 defect this
tree already carries. **The 23-suite classification is left to #408/#1509**, and
the measurement above is recorded here so that whoever takes it starts from a
number rather than from a re-derivation.

What is taken instead is the shape `REQUIRED_TESTS` already uses for C++: an
explicit pinned set, empty of judgement calls, that reds the moment a named
entry falls out of either lane. Seeded with one suite, it is a mechanism the
population row can later widen rather than replace.

## Design

### #1832

`tests/scripts/test_sglang_lease_identity.py` gains a module-level literal:

```python
EXPECTED_FILE_COUNT = 3338
```

with two new cases asserting `len(manifest["files"])` and
`manifest["file_count"]` against it separately, so that a mutation of either
half reds on its own name rather than through the other. The existing
self-consistency case stays: it is not wrong, only insufficient, and it is the
case that catches a header edited without the table.

**Exact, with no tolerance.** A wheel at a pinned revision has exactly one file
population. Any movement in it is either a new pin — which changes `pin`,
`wheel.sha256` and this literal together, in a change that says so — or a
defective manifest. There is no third case for a tolerance to admit, and a floor
below the real count would be the mute switch named above.

**The literal is in the test, not read out of the manifest.** This is the rule
`scripts/check-test-registration.py`'s own docstring states for
`REQUIRED_LABEL_SELECTIONS`: a checker that reads its expectation from the file
it checks is a tautology, which is precisely what #1832 found.

A third case binds the three records to the same literal. Each of
`.agents/environment.md`, `.agents/oracles/sglang.md` and
`.agents/sglang-matrix.md` quotes this count as measured; the case asserts the
decimal literal is present in each, so regenerating the manifest without
correcting the prose reds, and correcting the prose without the manifest reds.
It asserts *presence in the named file*, not a line number: an anchor that
counts lines in a file other people append to goes stale within the same PR.

### #1833

`scripts/check-test-registration.py` gains:

```python
REQUIRED_SUITE_REGISTRATIONS = {
    "test_sglang_lease_identity": "tests/scripts/test_sglang_lease_identity.py",
}
```

and `wiring_errors` iterates it, requiring each key in the preflight `SUITES`
array and each value as a direct active CI command. The existing hardcoded
checks for `check-test-registration` and `test_check_test_registration` are left
byte-for-byte alone: their diagnostics are the needles mutation cases M9–M12
match on, and rewording them to share a loop would move four pinned assertions
for no gain.

The reachability question is answered by the call chain that already exists:
`main` → `check_tree` → `wiring_errors(<real preflight>, <real ci>)`, and
`check-test-registration` is itself in preflight `CHECKERS` and in an
unconditional CI step. Nothing new needs wiring; the new rule rides the lane the
old rule already proved.

Two mutation cases join the fixed inventory, in the shape M9–M12 use:

- `M56` — the fixture's preflight `SUITES` entry deleted.
- `M57` — the fixture's CI command deleted.

`PASSING_PREFLIGHT` and `PASSING_CI` in
`tests/scripts/test_check_test_registration.py` grow the corresponding lines, so
`test_complete_preflight_and_ci_wiring_passes` remains a positive control that
is not vacuous. Adding the two names obliges
`tests/scripts/check_test_registration_mutations.txt` and the
`MUTATION_MANIFEST_SHA256` literal beside it; that coupling is the design of the
existing integrity layer, not a new one.

## Risks

- **Widening a red to green.** None of the changes removes or loosens an
  assertion. The existing `test_file_count_agrees_with_the_file_table` and the
  M9–M12 diagnostics survive unedited; every edit is additive.
- **A pinned literal going stale at the next pin.** `3338` is bound to
  `pin = f63458b5…`. Advancing the SGLang pin now reds three cases by name
  instead of silently accepting a manifest for a different tree — which is the
  behaviour being bought, not a cost.
- **The mutation-manifest sha256.** Editing the inventory without editing the
  literal reds `mutation_suite_integrity_errors`. The new sha256 is computed
  from the committed bytes and shown in `## Evidence`.
- **A concurrent edit to `sglang-wheel-in-lease.md`.** Another session is
  measuring on `dgx:gpu0` and owns that file's measurement sections. This row
  edits **one bullet** under `## Owed` and nothing else in it. If `origin/main`
  moves, the resolution takes main's complete file and re-applies that one
  bullet, proven with `git diff --numstat origin/main -- <path>`.

## Tests

| id | file | what it holds |
|---|---|---|
| T1 | `test_sglang_lease_identity.py::test_the_file_table_holds_the_pinned_population` | `len(files)` is `EXPECTED_FILE_COUNT` |
| T2 | `test_sglang_lease_identity.py::test_the_declared_file_count_is_the_pinned_population` | the `file_count` header is `EXPECTED_FILE_COUNT` |
| T3 | `test_sglang_lease_identity.py::test_the_records_quote_the_pinned_population` | the three records quote the same decimal |
| T4 | `test_check_test_registration.py::test_M56_deleting_the_pinned_preflight_suite_fails` | the preflight lane |
| T5 | `test_check_test_registration.py::test_M57_deleting_the_pinned_ci_suite_fails` | the CI lane |

T1, T2, T4 and T5 each have a recorded red-before on the unmodified base at
rc=0 with the mutation proven applied. T3 is red-before by construction: it is
asserted against the same literal T1 pins, and the mutation is the literal
moving.

## Gates

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/scripts/test_sglang_lease_identity.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check-test-registration.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/scripts/test_check_test_registration.py
scripts/agent-preflight.sh --fail-on-skip
```

Each must exit 0 on the landed tree, and each of the four mutations in
`## Evidence` must exit non-zero.

## Evidence

Filled in by the implementing commit. Every mutation carries a
`git diff --numstat` proving it applied, an exit code captured directly rather
than through a pipe, and a sha256 or `git status --porcelain` proving the tree
was restored.

## Stop conditions

- Stop and report `NEEDS_DECISION` if repairing either gate would require
  editing a measurement section of `sglang-wheel-in-lease.md`, changing
  `gateable = yes`, or taking a GPU lease.
- Stop if the symmetric population rule turns out to be unavoidable to make
  #1833 red. It is not: an explicit pinned set reds without classifying the 23.

## Now

`ACTIVE`. Spec committed; implementation follows in the same pull request.
