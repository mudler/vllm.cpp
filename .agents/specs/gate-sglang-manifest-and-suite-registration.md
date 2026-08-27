# Spec — two gates around the SGLang lease identity suite that measure nothing

Issue: [#1832](https://github.com/mudler/vllm.cpp/issues/1832),
[#1833](https://github.com/mudler/vllm.cpp/issues/1833)
Row: `GATE-SGLANG-MANIFEST-AND-SUITE-REGISTRATION` (unplaced gate defect; both
findings are properties of checkers, not of the `SGLANG-ORACLE-LEASE-WHEEL`
capability, which is `DONE`)
State: `DONE`

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

`3338` is quoted as measured in three records the issue names —
`.agents/environment.md`, `.agents/oracles/sglang.md` and the
`SGLANG-ORACLE-LEASE-WHEEL` row of `.agents/sglang-matrix.md` — and in two more
this row's fresh review found, `.agents/specs/sglang-wheel-in-lease.md` and
`docs/benchmarks/open-gaps.md`, while `grep -rn 3338 scripts/ tests/scripts/
.github/` exits 1. A number that no executing code holds is a number that
drifts, and this one is the denominator of `IDENTITY_RC=0`.

In scope: the count becomes an **assertion**, held as a literal in the checking
file and never read back out of the file it checks, and every record that states
the population is bound to that same literal so the prose cannot drift from it
either.

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

Two further cases bind the records to the same literal. Each of
`.agents/environment.md`, `.agents/oracles/sglang.md` and
`.agents/sglang-matrix.md` quotes this count as measured, so regenerating the
manifest without correcting the prose has to red, and correcting the prose
without the manifest has to red too.

**Every figure a record states, not one of them.** The first draft of this row
searched each record for any match of `(?<!\d)3338(?!\d)`, which is a
PRESENCE test, and presence is satisfied by one correct occurrence in a file
that has several. The fresh review measured the occurrences — one in
`../environment.md`, two in `../oracles/sglang.md`, three in
`../sglang-matrix.md` — and mutated the others green: `**3337 of 3338**` in the
oracle record, a manifest MISMATCH and the exact defect this gate exists to
catch, left the suite at `Ran 17 tests ... OK`, rc=0. `POPULATION_SLOTS` is
therefore a set of twelve phrase shapes that read a number AS this manifest's
population, and every capture of every shape is asserted against the literal.

**The swept set is derived, not listed.** A three-path allowlist cannot see a
FOURTH record quoting a wrong count, and there are five: the sweep of
`.agents/**/*.md`, `docs/**/*.md` and `*.md` also finds
`.agents/specs/sglang-wheel-in-lease.md`, whose recorded run transcript states
the population eight times, and `docs/benchmarks/open-gaps.md`. Any document
that states the population is bound the moment it says so.
`.agents/issue-index.md` is excluded by name because it is append-only and its
rows cannot be rewritten at the next pin; `QUOTING_RECORDS` is kept as the set
the sweep must still FIND, so a record that goes silent reds instead of
shrinking the swept population to nothing.

**The slots are manifest-specific, and that is a bound in both directions.** A
rule broad enough to read every `N files` fires on the 287-file and 81-file
populations two other rows measure; a near-miss band around 3338 fires on the
3335 and 3336 source-tarball figures the lease spec tabulates. Over all 886
markdown files in the tree the twelve shapes select 21 figures in five files and
nothing else. The separator is stripped before comparing, so `3,338` reads as
the correct count rather than as a defect — it reds under a bare-decimal search.

The cases assert *a phrase in a named file*, never a line number: an anchor that
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
- **The slot vocabulary is enumerated, and a NEW phrasing escapes it.** The
  twelve shapes are the shapes the tree uses. A record that states the
  population in a sentence none of them matches is not swept, and no checker can
  notice that. This is the residual of #1832 that survives the repair, and it is
  bounded the way the alternatives are not: the two rules broad enough to need
  no vocabulary — every `N files`, or a near-miss band — each fire on figures
  other rows measured correctly, which is a gate that reds on ordinary work.
  Adding a shape is one line.
- **`.agents/issue-index.md` is out of reach and states the population three
  times.** It is append-only by policy, GitHub holds the state of its rows, and
  a pin advance cannot rewrite them, so it is excluded by name. Its figures are
  correct today; the exclusion is a policy boundary and not a licence.
- **This spec is inside the swept corpus, and that shows in its own prose.**
  A wrong population figure written in slot shape reds the gate, including one
  quoted as the diagnostic of a mutation. The evidence table therefore quotes
  those diagnostics truncated (`'3337-file manif…'`), which is the cost of a
  rule that reads records rather than an allowlist, and it was measured by this
  spec redding the suite once while it was being written.
- **The sweep reads every markdown file in the tree.** 886 files, about 2.3 s,
  once per process behind an `lru_cache`. It is a read-only sweep of a corpus
  that is already in the checkout, and it is the price of not hardcoding a
  record set that cannot see a fourth record.
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
| T3 | `test_sglang_lease_identity.py::test_the_records_quote_the_pinned_population` | each of the three records still STATES a population figure, in the derived sweep |
| T3b | `test_sglang_lease_identity.py::test_no_record_states_a_population_other_than_the_pinned_one` | EVERY figure in EVERY swept record is `EXPECTED_FILE_COUNT` |
| T4 | `test_check_test_registration.py::test_M56_deleting_the_pinned_preflight_suite_fails` | the preflight lane |
| T5 | `test_check_test_registration.py::test_M57_deleting_the_pinned_ci_suite_fails` | the CI lane |

T1, T2, T4 and T5 each have a recorded red-before on the unmodified base at
rc=0 with the mutation proven applied. T3 is red-before by construction: it is
asserted against the same literal T1 pins, and the mutation is the literal
moving. T3b has six recorded reds in `## Evidence`, three of them the exact
mutations that were GREEN against the presence test T3 used to be.

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

Base `331eda8887e6a5c06244944c328b949b035cce4a`, implementation `db0fa4672`.
Every run sets `PYTHONDONTWRITEBYTECODE=1` and the tree is purged of
`__pycache__` first, because a restored file can still run a mutant's bytecode.
Every exit code below is `echo $?` on the command itself, never through a pipe:
`cmd | tail; echo $?` reports `tail`'s status.

### #1832 — the manifest population

| # | mutation | proof it applied | before | after |
|---|---|---|---|---|
| 1 | `files` emptied, `file_count` set to `0` | numstat `3 3342`; `file_count=0 len(files)=0` | `Ran 14 tests ... OK`, **rc=0** | `FAILED (failures=2)`, **rc=1** |
| 2 | `sglang/README.md` dropped, `file_count` set to `3337` | numstat `1 2`; `file_count=3337 len(files)=3337 README present=False` | `Ran 14 tests ... OK`, **rc=0** | `FAILED (failures=2)`, **rc=1** |
| 3 | `file_count` set to `3337`, table untouched | numstat `1 1` | already red | `FAILED (failures=2)`, **rc=1** |

Mutation 3 is the control that shows the kept self-consistency case still earns
its place: it reds `test_file_count_agrees_with_the_file_table` **and** the new
header case, which is the one shape the old case could see.

Both 1 and 2 restore to sha256
`5f6fdf983f084c44f8578645b2d97928a003d75cdf55863acb0ee91043e63a7e`, the
committed manifest.

Instrument proofs, so that none of the three new cases is vacuous:

- `EXPECTED_FILE_COUNT` moved to `3337` on the untouched manifest reds all
  three by name — table, header and records — `FAILED (failures=3)`, **rc=1**.
- Rewriting the occurrence pair in `.agents/oracles/sglang.md` to `XXXX`
  (numstat `1 1`) reds only `test_the_records_quote_the_pinned_population`, with
  `AssertionError: Lists differ: ['.agents/oracles/sglang.md'] != []`,
  **rc=1**. So the sweep is still held to the three NAMED records: a record that
  stops stating the population reds instead of dropping out of a derived set.

### The fresh review's findings on this repair, and the reds that answer them

The first implementation searched each record for any match of
`(?<!\d)3338(?!\d)`, which is presence, not agreement. The review measured the
occurrences — 1, 2 and 3 — and showed that in a record with more than one, the
others could be mutated green. These are the same mutations against the repair.
Each is applied to the working tree, proven applied, and restored from a
byte-copy afterwards, because `git checkout --` restores HEAD and would silently
discard the uncommitted repair being measured.

| # | mutation | proof it applied | against the presence test | against `POPULATION_SLOTS` |
|---|---|---|---|---|
| 7 | `.agents/sglang-matrix.md`, the first of three occurrences decremented | numstat `1 1` | `Ran 17 tests ... OK`, **rc=0** | `FAILED (failures=1)`, **rc=1**, naming `'3337 files of the ins…'` |
| 8 | `.agents/oracles/sglang.md`, `**3338 of 3338**` → `**3337 of 3338**` | numstat `1 1` | `Ran 17 tests ... OK`, **rc=0** | `FAILED (failures=1)`, **rc=1**, naming `'3337 of 3338…'` |
| 9 | `.agents/environment.md`, its single occurrence decremented | numstat `1 1` | `FAILED`, rc=1 | `FAILED (failures=1)`, **rc=1**, naming `'3337-file manif…'` |
| 10 | `docs/benchmarks/open-gaps.md`, a FOURTH record no allowlist held | numstat `1 1` | invisible, **rc=0** | `FAILED (failures=1)`, **rc=1** |
| 11 | `.agents/specs/sglang-wheel-in-lease.md`, `manifest_files=` in the run transcript | numstat `1 1` | invisible, **rc=0** | `FAILED (failures=1)`, **rc=1** |
| 12 | `.agents/environment.md`, its population phrase removed entirely | numstat `1 1` | — | `test_the_records_quote_the_pinned_population` **rc=1**, `Lists differ: ['.agents/environment.md']` |

Mutation 9 is the one the review reported as green on the strength of an
unrelated `.../issues/3338` URL sharing the line. **That is not what this tree
holds.** `.agents/environment.md` contains exactly one match of the decimal and
no such URL, and the mutation reds on the presence test as well as on the slot
rule. It is kept in the table as a measured negative.

The thousands separator is now handled rather than merely erring safe:
`3338-file` → `3,338-file` in `.agents/environment.md` (numstat `1 1`) is a
CORRECT statement of the count and reads `Ran 18 tests ... OK`, **rc=0**. It was
**rc=1** against the bare-decimal search.

Instrument proofs for the two new-shaped cases:

- `EXPECTED_FILE_COUNT` moved to `3337` (line 149 read back as `3337`) reds the
  table case, the header case and the slot sweep by name,
  `FAILED (failures=3)`, **rc=1**.
- `POPULATION_SLOTS` emptied to `()` reds
  `test_the_records_quote_the_pinned_population`, **rc=1**: a sweep that finds
  nothing cannot pass vacuously.

### #1833 — both registrations

`scripts/check-test-registration.py`, exit code captured directly:

| # | mutation | proof it applied | before | after |
|---|---|---|---|---|
| 4 | `  test_sglang_lease_identity` deleted from `SUITES` | numstat `0 1` | `OK: ...`, **rc=0** | `ERROR: pinned suite test_sglang_lease_identity is missing from preflight SUITES`, **rc=1** |
| 5 | the whole 11-line CI step deleted | numstat `0 11`; `yaml.safe_load` still parses | `OK: ...`, **rc=0** | `ERROR: pinned suite test_sglang_lease_identity is missing from the CI suite lane`, **rc=1** |
| 6 | both of the above at once | numstat `0 11` and `0 1` | not measured before | both errors, **rc=1** |

Instrument proof: deleting the five-line loop from `wiring_errors` (numstat
`0 5`) reds `test_M56_...` and `test_M57_...` and nothing else,
`FAILED (failures=2)`, **rc=1**. `scripts/check-test-registration.py` restores
to sha256 `655e9d2349d2d959f24fd2b5bf2667be357cf4bfe955fbd7af54c473afc678e1`.

**The scope control, recorded because it is deliberate.** Deleting the
unrelated `  test_tower_skip_rss_report` from the same array (numstat `0 1`)
still leaves the checker at **rc=0**. That is the 23-suite population rule left
to #408 and #1509, and it is measured here rather than left to be rediscovered.

### Green after

| command | result |
|---|---|
| `python3 tests/scripts/test_sglang_lease_identity.py` | `Ran 18 tests ... OK`, rc=0 (was 14; 17 before the review repair) |
| `python3 tests/scripts/test_check_test_registration.py` | `Ran 70 tests ... OK`, rc=0 (was 68) |
| `python3 scripts/check-test-registration.py` | `OK: ... together with 1 pinned suite(s) [test_sglang_lease_identity] in BOTH lanes.`, rc=0 |

`git status --porcelain` is empty after every mutation block.

## Stop conditions

- Stop and report `NEEDS_DECISION` if repairing either gate would require
  editing a measurement section of `sglang-wheel-in-lease.md`, changing
  `gateable = yes`, or taking a GPU lease.
- Stop if the symmetric population rule turns out to be unavoidable to make
  #1833 red. It is not: an explicit pinned set reds without classifying the 23.

## Now

`DONE`. Spec and implementation are in one pull request, the spec committed
first (`d6ce84b2a` before `db0fa4672`).

## Outcome

**Measured, and it changed the design.** The symmetric population rule was the
obvious repair for #1833 and it was rejected on a number: 11 preflight-only and
12 CI-only suites, read with the checker's own `_bash_array_values` and
`_active_ci_commands`, not estimated. Twenty-three classifications is a
different row, and the explicit pinned set gets the same red on the suite that
prompted the issue without pretending to make those twenty-three decisions. The
control on `test_tower_skip_rss_report` is left reading rc=0 on purpose and said
out loud, because a scope boundary that is not measured reads later as a gate
that failed.

**Rejected: a floor.** `len(files) >= some_number` was never a candidate. This
tree already carries `NORETURN_POPULATION_FLOOR = 40` against a real 53, which
is a mute switch for thirteen entries, and the manifest is the only identity
assertion this oracle has.

**Rejected: reading the expected count out of the manifest, or out of the
spec's recorded run transcript.** Both are the tautology #1832 names, one
document short. The literal is in the executing file and the records are
asserted against it, which is the direction that catches prose drift too.

**Why the records case asserts a phrase rather than a line.** `.agents/`
records are appended to by other rows; a line anchor recorded here goes stale
inside one pull request.

**Rejected, after the review measured it: presence of the decimal.** The first
implementation searched each record for `(?<!\d)3338(?!\d)`. It cannot be
satisfied by a longer number containing the digits, which was the property it
was chosen for, and it is still the wrong assertion: a record that states the
population more than once is answered by any ONE correct occurrence, so the
oracle record could state a manifest MISMATCH and stay green. Agreement of every
figure replaced presence of one, and the swept set is derived so that a fourth
record cannot sit outside it.

**Rejected: a near-miss band, and an every-`N files` rule.** Both need no
vocabulary and both fire on figures other rows measured correctly — 3335 and
3336 in the lease spec's source-tarball table, 287 and 81 in two other specs. A
gate that reds on ordinary work is the defect, not the discipline. The cost of
the narrow rule is an enumerated vocabulary, recorded in `## Risks`.

**What is NOT owed, corrected here.** This row's first draft repeated an
inherited framing: that the manifest is "pinned but not independently
re-derived" and owes a second independent install. Job `86282a1a` on 2026-08-23
already is one — a fresh `pip download` on another machine, four days after the
manifest was generated, hashing the INSTALLED tree with `derive()` rather than
reading the committed file table, by a different method from the zip-member
hashing that produced the manifest, and reporting `extra` as well as `missing`.
The correction is in the owed bullet of
[`sglang-wheel-in-lease.md`](sglang-wheel-in-lease.md). #1832 is left open for a
human to close rather than closed by this row's merge, because a squash message
cannot be taken back. #1265 stays open, `gateable = yes` is untouched, no
measurement section was edited, and no GPU was leased for this row.
