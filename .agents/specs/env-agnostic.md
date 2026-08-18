# ENV-AGNOSTIC-CAMPAIGN: make the tree readable from a second network

Issue: [#1190](https://github.com/mudler/vllm.cpp/issues/1190).
Row: `ENV-AGNOSTIC-CAMPAIGN`. This is the scoping row for a campaign, so it
lands the rule, the mechanism, and one worked example. It does not sweep the
tree, and the waves that do are defined under `## Follow-on rows`.

## Scope

One operator's host names, share paths, and network addresses are written into
tracked files where a resolved placeholder belongs. A second developer who
follows the protocol documents is told to reach a box that exists on one home
network. That is a portability defect in the documentation.

In scope for the campaign: every tracked file outside `.agents/completed/`.

Out of scope, permanently, and each for its own reason:

- `.agents/completed/`, 18 files and 107 hits. The archive records what
  happened.
- `.agents/issue-index.md`, 1 file and 24 hits. The file is append-only under a
  `merge=union` driver, so an edited row is duplicated rather than merged.
- `docs/bench-evidence/`, 2 files and 2 hits. Captured evidence.
- Every literal that `## The classification rule` calls provenance, wherever it
  lives. Rewriting a recorded measurement's host falsifies the record.

## The derivation query

Re-run this instead of quoting a number from this file. The counts move with
every landing, and a quoted number becomes treated as measured.

```sh
git grep -cIE 'dgx\.casa|nas_share|192\.168\.|thor:gpu0' -- . | sort -t: -k2 -rn
git grep -lIE 'dgx\.casa|nas_share|192\.168\.|thor:gpu0' -- . | wc -l
```

Add `':!.agents/completed'` as a final pathspec to exclude the archive. Swap
`-c` for `-l` to count files and for `-ho` to count hits.

Measured at base `fd64c76ee45ba49b070ea83024f6678ddd7f64a6` on 18 August 2026:

| Pattern | Files | Hits |
|---|---:|---:|
| `dgx.casa` | 203 | 620 |
| `nas_share` | 32 | 83 |
| `192.168.` | 24 | 74 |
| `thor:gpu0` | 11 | 46 |
| Union | 227 | n/a |
| Union outside `.agents/completed/` | 209 | n/a |

`Files` comes from `git grep -l` and `Hits` from `git grep -ho`, so `Hits`
counts matches and not matching lines. The issue reported 200 files for
`dgx.casa` and 18 hits for `.agents/environment.md`. Both are now higher, at
203 files and 34 matches.

## The classification rule

Two classes, and one test that separates them.

**Configuration.** A host, path, device, or address that a reader must
substitute to follow an instruction. It becomes a `${KEY}` placeholder that
resolves from `.env`.

**Provenance.** A host, path, device, or address that names where a recorded
observation happened, or that defines a named environment profile. It stays
literal.

**The substitution test.** Replace the literal with a second developer's value
and read the sentence again. If the sentence stays true for that reader, the
literal is configuration. If the sentence becomes false, the literal is
provenance, because the sentence is about that specific machine.

Worked pairs from the tree at the base SHA:

| Sentence | Class | Why |
|---|---|---|
| `ssh dgx.casa` then `cd ~/work/vllm.cpp` (`scripts/dgx-bringup.sh:10`) | configuration | Another host and checkout run the same gate, so the instruction stays true |
| "`/workspace` is the house NAS, measured as `//192.168.68.102/Data 7.3T total`" (`.agents/environment.md:99`) | provenance | Another address was not what the probe read, so the sentence becomes false |
| "**Ettore DGX release-gate profile**: device `dgx:gpu0`, host `dgx.casa`" (`.agents/environment.md:360`) | provenance | The literal is the profile's definition and not a parameter of it |
| "`thor:gpu0` read `unknown (no contact 1m0s)` on 2026-08-17" (`.agents/environment.md:35`) | provenance | A dated reading of one device |
| "the pinned oracle venv lives at `~/venvs/vllm-oracle-pin-555967922`" (`.agents/environment.md:105`) | configuration | `VLLM_ORACLE` already exists for exactly this value |

Three grammatical signals make the test fast. A past-tense verb with a date, a
run identifier, or a measured quantity is provenance. An imperative verb, a
copyable command, or a prerequisite list is configuration. A heading that names
a person or a profile makes every literal under it that profile's definition.

**A blind `sed` corrupts the second class.** The two classes share every
literal, so no pattern can separate them. This is why the campaign is a reading
task with waves and reviewers, and why no checker enforces it.

### The registry is not drift

`.agents/environment.md` is a registry of named environment profiles, and its
own instructions at `:340` invite a second developer to add a profile entry in
the same shape. Every one of its 34 hits is provenance or profile definition
under the test above, so the densest guide file needs no substitution at all.
`AGENTS.md` names `dgx:gpu0`, `thor:gpu0`, and `orin:gpu0` for the same reason,
and it already records that `rc devices` is the live membership list and that
the written three are a lower bound. Leave both.

That result reverses the issue's ranking. Density does not predict the defect.

## The mechanism already exists and was almost unused

`.env.example` is tracked, `.env` is ignored at `.gitignore:31`, and
`.agents/developer-preferences.md` is ignored at `.gitignore:27`.
`scripts/agent-onboard.py` already detects a missing, incomplete, or unreadable
`.env`, and `--env-set KEY=VALUE` already records one answered value and
refuses any key `.env.example` does not declare.

Placeholder use of the twelve keys at the base SHA, counted with
`git grep -lI '${KEY'`:

| Key | Files using `${KEY}` |
|---|---:|
| `VLLM_SOURCE` | 59 |
| `GPU_LOCK` | 50 |
| `VLLM_ORACLE` | 6 |
| `DEPENDENCY_SOURCE` | 3 |
| `CUTLASS_DIR` | 3 |
| `GATE_HOST` | 1 |
| `CHECKPOINT_ROOT` | 1 |
| `SGLANG_SOURCE`, `LLAMACPP_SOURCE`, `DEVICE_ARCH`, `DEVICE_TOOLKIT_ROOT`, `DEVICE_COMPILER` | 0 |

`VLLM_SOURCE` and `GPU_LOCK` show that the mechanism works when it is applied.
The literals the campaign has to convert are mostly values that already have a
key. `~/venvs/vllm-oracle` appears in 108 files while `${VLLM_ORACLE}` appears
in 6, `/usr/local/cuda` appears in 44 while `${DEVICE_TOOLKIT_ROOT}` appears in
none, and `cutlass-4.5.0` appears in 37 while `${CUTLASS_DIR}` appears in 3.

### The three keys the tree needs and lacked

Derived from the literals that recur and map to no existing key:

| New key | Literal it replaces | Files | Hits |
|---|---|---:|---:|
| `GATE_CHECKOUT` | `~/work/vllm.cpp` | 59 | 370 |
| `SHARED_STORAGE_ROOT` | `/usr/local/nas_share`, `/mnt/nas_share`, `/workspace` | 51 | 148 |
| `GATE_DEVICE` | `dgx:gpu0`, `thor:gpu0`, `orin:gpu0` | 22 | 112 |

Counts exclude `.agents/completed/`. `GATE_DEVICE` is deliberately singular. It
names the one device this developer's gates use, and `rc devices` stays the
enumeration, so the tree never carries a fleet list that goes stale.

No key was added for a second or third profile host. `.env` carries the one
selected environment, and `.agents/developer-preferences.md` selects which
profile that is. A `GATE_HOST_2` would put the profile registry back into the
tracked tree.

## Design

Three changes land here.

1. `.env.example` gains `SHARED_STORAGE_ROOT`, `GATE_DEVICE`, and
   `GATE_CHECKOUT`, each commented and copy-paste usable. The file stays the
   only source of legal keys, so `agent-onboard.py` picks them up with no
   code change.
2. `AGENTS.md` gains the create-on-first-use obligation in `## Start here`,
   next to the never-infer rule it completes.
3. `scripts/agent-start.py` gains the route that carries that obligation to the
   session that needs it.

### Why the ask path lands in both the policy and the router

`AGENTS.md` already forbids inference and already tells an agent to resolve
both untracked files. Prose alone did not close the gap, because the router
printed `environment: missing` as a status label and then listed next actions
that never mentioned it. A session read an unresolved environment and had no
instruction, and the instruction it reached for in practice was a host name
copied out of a document.

The router is the right place for the ask because `scripts/agent-start.py` is
step 1 of the protocol and every session runs it. It stays read-only and
non-interactive, which its own module docstring requires. It detects, it names
the recording command, and it never writes, because no harness-neutral
mechanism lets a script hold a conversation. `scripts/agent-onboard.py` owns
the write and already refuses an undeclared key.

The policy is the right place for the obligation because a checker cannot see
whether an agent asked. `AGENTS.md` states what must happen, the router makes
it visible at the moment it applies, and the two do not restate each other.

The route never names which key is unset. That set is the caller's own state
and can carry a secret, and `tests/scripts/test_agent_start.py` already pinned
that property before this row.

## The worked example

`scripts/dgx-bringup.sh`. It was chosen after `.agents/environment.md` was
measured and rejected, and after `.agents/coordination.md` and
`.agents/parity-ledger.md` were read as claim and measurement records.

`scripts/dgx-bringup.sh` is the one tracked file that gives an unconditional
`ssh dgx.casa` instruction outside any profile. At the base SHA it carried 16
resolved literals in 52 lines, counted with

```sh
git show origin/main:scripts/dgx-bringup.sh \
  | grep -ohE 'dgx\.casa|/usr/local/cuda|121a|cutlass_probe|~/work/vllm\.cpp|~/venvs/vllm-oracle|~/work/apex|~/\.cache/huggingface' \
  | sort | uniq -c
```

which reports `121a` 4 times, `dgx.casa` 4, `~/.cache/huggingface` 2,
`cutlass_probe` 2, and `/usr/local/cuda`, `~/venvs/vllm-oracle`, `~/work/apex`,
and `~/work/vllm.cpp` once each. Every one is configuration under the test. The
file also has a prose header and an executable body, so it proves both halves
of the pattern at once.

The conversion found a live defect. The script defaulted `CUTLASS_DIR` to
`${HOME}/cutlass_probe`, while `.agents/environment.md:389` records
`-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0` as mandatory on the same box. The
script's default had already drifted from the box it is named after. A
configure that does not find CUTLASS drops the sm120a NVFP4 GEMM and
FlashAttention-2 without saying so, and `.agents/environment.md:388-400` records
that turning those off moves the SACRED `test_qwen27_paged_engine` from 235/235
to 234/235 with the source untouched. A stale hard-coded default is therefore a
false green and not an inconvenience.

The pattern the waves reuse:

1. Read every literal and apply the substitution test.
2. Replace a configuration literal with `${KEY}` in prose.
3. Resolve a configuration literal in code from the process environment first
   and the repository `.env` second.
4. Refuse loudly when a required value is unset. Name the key, name the file,
   and refuse before any expensive step.
5. Never write a fallback value. `${KEY:-}` saves an answer and supplies
   nothing, while `${KEY:-literal}` is the defect this row removed.
6. Leave the file path alone. `.agents/completed/` references
   `scripts/dgx-bringup.sh` by name, so a rename falsifies an archived record.

The shell wins over the file, and that ordering needed code. A `.env` copied
from `.env.example` declares every key blank, so `set -a; . ./.env; set +a`
assigns the empty string over a value the caller exported. The script saves the
shell's answers, sources the file, then restores them.
`tests/scripts/test_gate_bringup.py` found that defect, and it is the reason
the process-environment contract in `.env.example` is now executable rather
than documented.

## Follow-on rows

Each is independently landable, owns a disjoint file set, and needs its own
issue, spec, fresh implementer, and fresh reviewer. Waves 1 to 3 convert. Wave
4 classifies and is expected to change almost nothing.

| Row | Owns | Files | Hits | Shape |
|---|---|---:|---:|---|
| `ENV-AGNOSTIC-W1-TOOLING` | `scripts/`, `tools/`, `third_party/README.md`, minus `scripts/dgx-bringup.sh` | 19 | 22 | Convert, and add a refusal test for each script that requires a value |
| `ENV-AGNOSTIC-W2-GUIDES` | `AGENTS.md`, `.agents/environment.md`, `.agents/benchmarking.md`, `.agents/porting.md`, `.agents/bugfixing.md`, `.agents/workflow.md`, `.agents/verification.md`, `docs/USAGE.md`, `README.md` | 4 | 38 | Convert, and confirm the profile registry and the fleet-device names stay |
| `ENV-AGNOSTIC-W3-CODE` | `tests/`, `src/`, `include/`, `examples/`, `benchmarks/`, `.github/` | 48 | 65 | Convert comment and docstring recipes, and leave a golden's recorded capture host |
| `ENV-AGNOSTIC-W4-RECORDS` | `.agents/specs/`, `.agents/claims/`, the five area matrices, `.agents/porting-inventory.md`, `.agents/roadmap_v1.md`, `docs/superpowers/` | 129 | 372 | Classify only, and report how many hits are configuration |
| `ENV-AGNOSTIC-W5-LEDGERS` | `.agents/benchmark-record.md`, `.agents/parity-ledger.md`, `.agents/coordination.md`, `docs/STATUS.md`, `docs/BENCHMARKS.md` | 5 | 189 | Classify only, one reviewer per file, and expect no substitution |

Counts come from the derivation query at the base SHA. The five row file sets,
this row's one converted file, and the three permanently out-of-scope sets
partition all 227 files with no overlap and no remainder. Re-derive them at
each wave's own base.

Ordering: W1 and W2 first, because they carry the onboarding harm the issue
names. W3 next. W4 and W5 last, because a wave that is expected to change
nothing must not run before the rule has been applied where it does change
something.

W5 is deliberately last and deliberately small. A reviewer who reads the 107
provenance hits in `.agents/benchmark-record.md` and reports zero substitutions
has done the work correctly.

## Tests

- `tests/scripts/test_agent_start.py::test_unresolved_environment_routes_to_ask_and_record`
  requires the ask-and-record route on the declared and undeclared routes, for
  each of `missing`, `incomplete`, and `unreadable`, and requires that no unset
  key name is echoed.
- `tests/scripts/test_agent_start.py::test_resolved_environment_adds_no_ask_route`
  requires the route to be absent when `.env` is complete.
- `tests/scripts/test_gate_bringup.py` requires the bring-up script to refuse
  each required key by name before any configure step, to resolve values from
  the untracked `.env`, to let the process environment win over the file, and
  to carry no resolved literal in its executable half.

## Gates

- `python3 tests/scripts/test_agent_start.py`
- `python3 tests/scripts/test_gate_bringup.py`
- `python3 tests/scripts/test_agent_onboard.py`
- `scripts/agent-preflight.sh --fail-on-skip`

No hardware gate applies. This row changes no kernel, no dtype, no allocation,
and no token, so it claims no CUDA, SACRED, or throughput gate.

## Stop conditions

Stop and report `NEEDS_DECISION` when a literal passes the substitution test in
one sentence and fails it in the next sentence of the same paragraph. Split the
paragraph or leave both literal. Do not guess.

Stop and report `NEEDS_DECISION` before renaming any file whose name carries a
host, because `.agents/completed/` references file names and a rename falsifies
an archived record.

Never make a red gate green by widening a checker's scope. This campaign adds
no checker, and `## The classification rule` says why.

## Evidence

Red before green, `scripts/agent-start.py`:

```text
$ python3 tests/scripts/test_agent_start.py
Ran 22 tests in 0.205s
FAILED (failures=6)
```

Six failures, all `AssertionError: 'NO ENVIRONMENT' not found`, across three
`.env` states and two routes. Green after the change at 22 tests.

Mutation, `scripts/dgx-bringup.sh`. Reintroducing
`DEVICE_ARCH="${DEVICE_ARCH:-121a}"` above the refusal took
`tests/scripts/test_gate_bringup.py` from 4 passing to 2 failures. The tree was
restored and compared byte-for-byte, and the suite returned to green.

The router's real output on a worktree with no `.env`:

```text
NO ENVIRONMENT: .env is missing.
  Ask the developer for the ONE value the current gate needs,
  then record it, and leave every other key empty:
  scripts/agent-onboard.py --env-set KEY=VALUE
  An unanswered key stays empty and its gate stays PENDING.
  Never infer a value, and never take a host or a path from a
  document instead of asking.
```

`scripts/agent-preflight.sh --fail-on-skip` was run twice on this branch, and
the difference between the two runs is the host's load rather than the tree.

At loadavg 46 it reported 79 gates `ok`, zero `SKIP`, and one `FAIL`. The
failure is `test_cpu_x86_llamacpp_floor`, whose contended-leg case is
load-dependent and is tracked by
[#618](https://github.com/mudler/vllm.cpp/issues/618). The harness exited
`NO_QUIET_WINDOW` (4) rather than `GIVING_UP` (2), so the guarantee went
untested. Rerun serially on the same tree, that suite reported 10 tests `OK`.

At loadavg 12.9 the same command reported 80 gates `ok`, zero `FAIL`, zero
`SKIP`, and `All gates green.` Both committed-range blocks executed in both
runs, and all five of their gates are `ok` in both. This row touches no CPU
backend path.

A linked worktree never has `.env`, because the file is untracked and lives in
the checkout it was created in. Every worktree therefore sees this route. That
is correct rather than noisy, since a gate command run from a worktree resolves
nothing today.

## Owed

Nothing. Every gap this row found is owned by a follow-on row in
`## Follow-on rows`, and each of those rows opens its own issue at claim time.

## Outcome

Recorded when the campaign closes. The result to record first is whether
`## The classification rule` survived contact with `.agents/specs/`, and how
many of W4's 372 hits turned out to be configuration.

## Now

`ENV-AGNOSTIC-CAMPAIGN` is the scoping row and lands the rule, the three new
keys, the create-on-first-use route, and one converted file. The tree sweep is
`ENV-AGNOSTIC-W1-TOOLING` through `ENV-AGNOSTIC-W5-LEDGERS`, none of which is
claimed.
