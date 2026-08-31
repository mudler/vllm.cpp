# `ENG-PREFLIGHT-COMPILES` — compile what a change can break, before the push

Issue: [#2401](https://github.com/mudler/vllm.cpp/issues/2401).

| Section | Content |
|---|---|
| Scope | IN: `scripts/check-tree-compiles.py`, its suite, and its preflight and CI wiring. OUT: every policy edit to `AGENTS.md` (argued below, and left to the developer), linking, running tests, every non-default configuration, every CI build job, `.githooks/pre-push`, and what preflight demands of a `read-only` session |
| Upstream chain | NO vLLM analogue. This is local protocol machinery, so the mirror rule does not apply and there is no upstream `file:line` to port. Governed by `AGENTS.md` §"Changing the rules or a checker", which requires a spec, a red-before test or mutation, and green-after evidence. The nearest in-tree precedents are `.agents/specs/gate-prepush-fail-loud.md` (a gate that presented as six checks while running three) and `.agents/specs/gate-preflight-skip-report.md` (the three-state `ok`/`FAIL`/`SKIP` protocol this checker plugs into) |
| Our baseline | `scripts/agent-preflight.sh` runs 30 record checkers and 60 Python suites and compiles nothing. `.githooks/pre-push` runs three of the same checkers and compiles nothing. `main` was pushed twice on 2026-08-31 in a state that does not build, green on every one of them: `5263ac31f` and `08fa2f5aa` (#2395) |
| Port map | Nothing is ported. `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS`, already `ON` at `CMakeLists.txt:38`) → the exact per-TU flags; `c++ -MM -MG` → the exact reverse-include map; `c++ -fsyntax-only` → the front-end-only compile. Preflight's existing `run()`/`skip()` pair → the `ok`/`FAIL`/`SKIP` reporting |
| Tests to port | Nothing to port; `tests/scripts/test_check_tree_compiles.py` is written against this tree. The historical red-before trees are `5263ac31f^` and `08fa2f5aa^`, which pass every existing gate and must fail this one |
| Gates | `python3 tests/scripts/test_check_tree_compiles.py`; `python3 scripts/check-tree-compiles.py --base origin/main`; `scripts/agent-preflight.sh` |
| Dependencies | Row IDs: none blocking. `GATE-PREPUSH-FAIL-LOUD` owns the hook's checker list and is untouched; `GATE-PREFLIGHT-SKIP-REPORT` owns the three-state protocol and is reused, not changed. Toolchain: `cmake` ≥ 3.24, Ninja, a host C++20 compiler. Hardware: none; no GPU lease is taken |
| Work breakdown | (1) the spec and its records, committed first; (2) `tests/scripts/test_check_tree_compiles.py`, red before any implementation; (3) `scripts/check-tree-compiles.py`; (4) the preflight block and `SUITES` entry, plus the CI script lane; (5) the red-before/green-after evidence on the two historical trees. No `AGENTS.md` edit: the reason is in Design |
| Risks/decisions | DECISION — no cap on the affected-TU count: a cap below the real fanout is a mute switch, and the worst case measured over 60 commits is 783 TUs / ~3.1 min, still 4x cheaper than the build it replaces. DECISION — exit 2 maps onto `SKIP` rather than `FAIL`, because a box without a compiler is not a defective change, and preflight already refuses the green banner over a skip. RISK — a stale `compile_commands.json` would compile the wrong file; answered by configuring fresh into a scratch directory every run, which costs 1.67 s. RISK — parallel agent builds have OOM-killed this box; `-fsyntax-only` writes no object file and jobs default to half the CPU count capped at 8 |

## Scope

On 2026-08-31 `main` was pushed twice in a state that does not compile, and
`scripts/agent-preflight.sh` was green both times.

| Fix | Defect | Shape |
|---|---|---|
| `5263ac31f` | `tools/bench/ltx2_connector_gemm_probe.cpp` ended seven `//` comment lines with a shell continuation, which `-Wcomment` plus this tree's `-Werror` rejects | one translation unit, and the diff that introduced it named that file |
| `08fa2f5aa` | `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304` passed `MlaSharedSelection*` into `ForwardMlaAttentionBlock`'s `vt::Tensor*` parameter (#2395) | one translation unit, and **no contributing diff named it** |

Preflight runs 30 record checkers and 60 Python suites. Not one of them compiles
a translation unit. `check-commit-trailers`, `check-commit-style`,
`check-agent-record`, `check-symbol-anchors`, `check-env-doc`,
`check-attention-rung-consistency` and `check-issue-index-append-only` all
validate records, prose, anchors and trailers against a tree, and every one of
them passes on a tree that does not build.

`.githooks/pre-push` runs three of those record checkers against the pushed
commit's worktree. It compiles nothing either.

CI compiles. `build-test-cpu`, `build-newest-gcc`, `cuda-fat-build` and the
sanitizers would have caught both. They land a verdict up to two hours after the
push, and both defects reached `main` by a direct push, so the verdict arrived
after the damage. The gap is not "does anything build". It is "does anything
build BEFORE the push".

**The second defect fixes the shape of the answer.** Neither contributing commit
carried the broken call. `e799f7d2c` added `attn_pre_o_proj` to the header;
`ee5c86031` added the test that calls it. Each side compiled alone. A gate that
compiles "the `.cpp` files this diff names" sees the first defect and is blind to
the second. The scope has to follow `#include` edges out of the changed headers.

Out of scope: linking, running tests, any configuration other than the default
Linux host build, and the CI lanes, which keep their job unchanged.

## Design

One checker, `scripts/check-tree-compiles.py`, invoked from preflight.

**1. Resolve the scope, and say what it is.** Changed paths come from the union
of four components: `git diff --name-only <base> <head>`, the staged diff, the
unstaged diff and the untracked files. `<base>` defaults to `origin/main`
resolved to a commit, `<head>` to `HEAD`. Each component's path count prints, so
a reader can see which of the four carried the change, and a brand-new `.cpp`
that is not committed yet is in scope rather than invisible -- which is the
shape `5263ac31f` had while it was being written.

**2. An empty scope is derived, not skipped.** When no C++ source, header, or
build file is in scope, the affected-TU set is provably empty, the checker says
so in words with the range it read, and exits 0. It does not print a compile
count it did not earn. This is the arm that keeps the gate off the 30 of the last
60 commits on `main` that touch no C++ at all.

**3. Configure, do not build.** `cmake -S . -B <dir> -G Ninja` writes
`compile_commands.json` and nothing else: measured 1.7 s and 14 MB on this box.
That file carries the exact flags CMake would use for every TU, `-Werror`
included, so the checker cannot drift from the build by reconstructing flags of
its own.

**4. Follow the include edges, exactly.** When a header or build file is in
scope, the checker runs `c++ -MM -MG` over every TU in `compile_commands.json`
with that TU's own flags, and inverts the result into a header → TUs map. This is
the real preprocessor with the real include path, not a textual `#include` scan,
so a conditional include, a macro-formed path and a generated header all resolve
the way the compiler resolves them. Measured 15.9 s for all 1218 TUs at `-j8`
under a load average of 28. The scan is skipped entirely when only `.cpp` files
are in scope, because then the affected set is exactly those files.

A TU whose scan FAILS has no edges, and reading "no edges" as "reaches no
header" would drop it from the closure of every header change in silence. Those
TUs are named and checked unconditionally instead, so the compiler says what is
wrong with them rather than the scan swallowing them.

**5. Compile with `-fsyntax-only`.** The recorded command minus `-c` and `-o`,
plus `-fsyntax-only`. The front end runs, `-Werror` applies, and no object file
is written, so the check costs no disk and cannot collide with another agent's
build directory. Measured 0.27–3.26 s per TU, mean 1.76 s, on a loaded box.

**6. Three exit codes, and the third is never a pass.** `0` every TU in scope
compiled, or the set was provably empty. `1` at least one TU failed, and the
compiler's own message prints verbatim under the TU's path. `2` the check could
not run: no `cmake`, no compiler, a configure failure, an unresolvable base, or
no `compile_commands.json`. Preflight maps `2` onto its existing `SKIP` state
with the checker's reason, which already denies the "All gates green." banner and
already exits 1 under `--fail-on-skip`. Unknown is not absence and not success.

**7. Un-run is not passing, and it is not failing either.** A C++ source in
scope that no target compiles in this configuration -- a `.cu` behind an OFF
option, an orphaned file -- is printed BY NAME on its own line and is not added
to the compiled total, and a run whose attempt count does not equal its affected
count exits 2. On the `08fa2f5aa^` range that line names five `.cu` files and
two Tenstorrent units, so limit 2 is visible per run and not only in this
document. This is the distinction a `681 of 685 tests failed` summary could not
make when `-k 0` went to `cmake` instead of `ninja` and nothing had been
compiled at all.

### Why not the obvious shapes

**Not a full build in preflight.** ~12 minutes warm, 9.4 GiB, and it fires on
every records-only change. AGENTS.md §Gates: a gate that fires on ordinary work
is the defect. The per-class line budgets were retired for exactly this.

**Not "compile the changed `.cpp` files".** That is the cheap diff scope, and it
is blind to the second defect, which is the one this row exists for.

**Not a textual `#include` graph.** It over-approximates on `#if`-guarded
includes and under-approximates on generated and macro-formed ones, and it would
have to be maintained against the include path. `-MM` answers the same question
with the compiler, in 15.9 s.

**Not a new CI job.** CI already compiles four ways. Adding a fifth moves no
verdict earlier than the push.

**Not prose alone, and no AGENTS.md edit either.** A documented obligation to
build before pushing is a legitimate answer to this row and it is the wrong one
here, for a reason the two failures make checkable: preflight WAS run on both
trees. Both fix commits say so and name the gates that passed. The obligation
that was missing is not "run the gate"; it is that the gate had nothing in it
that compiles. AGENTS.md §Start here already says to run
`scripts/agent-preflight.sh` before editing, and §Landing work already says to
run the applicable gate before every push. Putting the compile inside preflight
makes both existing sentences true without writing a new rule, and a rule added
here would be a rule about a full build that costs 12 minutes -- which is
precisely the obligation that does not get paid.

So this row changes **no** policy prose. If the developer wants the obligation
stated as well as implemented, that is their call to make and this spec is the
proposal, not the edit.

### The known diff-scope failure mode, and what is done about it

`.agents/specs/` records that a diff-scoped checker can SKIP when its base moves
and still exit 0 — a green that covered less than the reader thinks. Three
things keep that out of here.

- The empty-scope arm prints the base SHA, the head SHA and the derived TU count
  every time, so a zero is visible as a zero rather than inferred from silence.
- An unresolvable base is exit `2`, not exit `0` with an empty diff. A base that
  cannot be read is the `CANNOT-VERIFY` arm.
- The scope is a union with the staged, unstaged and untracked diffs, so a moved
  base cannot empty the set for work in hand.
- The committed component is `git diff <base> <head>`, TWO dots, and that is
  deliberate. Two dots is symmetric, so when `origin/main` moves ahead the files
  main changed enter the scope as well. Three dots (`base...head`) would restrict
  it to the branch's own side and read smaller and cleaner -- and would be blind
  to exactly the `08fa2f5aa` shape, where main's header change is what breaks the
  branch's code.

**That symmetry has a measured cost, and it is the opposite of a silent skip.**
Run on this row's own branch, which changes no C++ file at all, after `main` had
moved ahead: 30 changed paths, 11 C++ sources, 39 translation units, 65 s. The
gate got wider as the branch fell behind, not narrower. A reader sees it,
because every one of those counts prints. The repair is to merge `main`, which
is what a branch that far behind owes anyway -- and it is confirmed rather than
asserted: rebasing this branch onto `origin/main` took the same command from
39 units and 65 s back to **0 units and 0.11 s**, with no change to the branch's
own diff.

## Tests

`tests/scripts/test_check_tree_compiles.py`, registered in preflight's `SUITES`
and in the CI script lane. One case per guarantee, each red before the checker
exists:

1. A TU in scope that does not compile exits 1 and prints the compiler's message.
2. A changed header pulls in a TU that includes it and that the diff does not
   name. This is the second defect's shape, and it is the case a changed-`.cpp`
   scope fails.
3. An empty scope exits 0, says the set is empty in words, and does not print a
   compiled count.
4. An unresolvable base exits 2, not 0.
5. A missing `compile_commands.json` exits 2, not 0.
6. The report names the base, the head and the TU count, so the instrument states
   what it compared.
7. Preflight maps exit 2 onto `SKIP` and exit 1 onto `FAIL`, executed rather than
   grepped.
8. A TU that does not preprocess is checked anyway. It exists at the base, is
   not in the diff, and the diff is a header edit that breaks nothing, so the
   only route to a non-zero exit is the forced inclusion.
9. Preflight runs the gate EXACTLY ONCE. Executed against a scratch `scripts/`
   that contains the checker, so the discovered `scripts/check-*.py` sweep has
   something to find; the stub `python3` logs every invocation and the case
   counts them.

Mutations, each applied to a scratch copy and restored under a sha256 check:
delete the reverse-include closure and case 2 must fail; turn the exit-2 arm
into exit 0 and cases 4 and 5 must fail; drop `-Werror` from the reconstructed
command and case 1 must fail; fold the uncompiled sources into the compiled
total and the un-run case must fail; drop the forced inclusion and case 8 must
fail; delete preflight's call site and case 7 must fail; drop
`check-tree-compiles.py` from preflight's `NAMED_CHECKERS` and case 9 must fail.

## Risks

- **The checker is itself a TU consumer.** If `compile_commands.json` goes stale
  against a moved source tree, the checker compiles the wrong file. Mitigated by
  configuring fresh into a scratch directory on every run; 1.7 s buys that.
- **Parallelism on a shared box.** `-fsyntax-only` allocates far less than a
  codegen-and-link job, and jobs default to half the CPU count capped at 8.
  Parallel agent builds have OOM-killed this box before.
- **The gate must not spawn a second copy of itself, and it did.** Preflight's
  discovered `scripts/check-*.py` sweep runs every name absent from
  `NAMED_CHECKERS`, so the first wiring executed this gate TWICE: once from its
  own block with a base, and again bare from the sweep. For the four names
  already on that list a second run is a cheap argparse usage error; here it was
  a second full `-fsyntax-only` pass at `-j8` -- 65 s, 39 units, 531 MB RSS --
  begun as the first finished, on a box at load average 157. The preflight run
  that found it died at exactly that point without writing an exit line. The
  name is now on the list and `test_preflight_runs_the_compile_gate_exactly_once`
  counts the invocations rather than reading the list.
- **A wide header change costs minutes, on EVERY preflight run.** `3bfd1a738`
  touched a core `vt` header and reaches 783 TUs; a whole-campaign range reaches
  789 and took 4.1 min measured. That is the correct answer for that change and
  it is still cheaper than the build it replaces, but preflight runs more than
  once per branch and pays it again each time. No cap is imposed, because a cap
  below the real fanout is a mute switch. The bounded fix is a result cache
  keyed on each unit's dependency-set hash and its recorded command, which would
  reduce a repeat run to the dependency scan; it is named under `## Owed` and is
  not built here.

## Gates

- `python3 tests/scripts/test_check_tree_compiles.py`
- `python3 scripts/check-tree-compiles.py --base origin/main`
- `scripts/agent-preflight.sh`

## Evidence

Measured on this box, 20 cores, with concurrent agent builds running throughout:
load average ~28 for the microbenchmarks and ~62 for the two historical runs.
Every figure is therefore pessimistic, and the two that matter most were taken
under the worse of the two.

| Quantity | Value |
|---|---|
| `cmake` configure, Ninja, Release | 1.67 s, 14 MB |
| TUs in `compile_commands.json` | 1218 |
| `c++ -MM -MG` per TU | 0.02-0.10 s |
| whole-tree dependency scan at `-j8` | 15.9 s |
| `c++ -fsyntax-only` per TU | 0.27-3.26 s, mean 1.76 s over 10 sampled TUs |
| full `cmake --build`, warm | ~12 min, 9.4 GiB |

**The two costs the row is judged on**, both measured end to end with the gate
as it ships:

| Change shape | Command | Cost |
|---|---|---|
| records-only (this branch: 8 paths, no C++) | `check-tree-compiles.py --base origin/main` | **0.20 s**, no configure, no compiler |
| a code change (a row's worth of `.cpp` and `.h`: 40 TUs) | `--base 76fefdca7 --head 5263ac31f` | **25.5 s** (1.4 s configure + 12.7 s scan + 11.3 s compile) |

Affected-TU fanout over the 60 commits ending at `9fa3be388`, which is what
makes the first number the common one:

| TUs affected | Commits | Gate cost |
|---:|---:|---|
| 0 | 30 | ~0.2 s, no configure |
| 1-4 | 20 | 3-22 s |
| 18-37 | 6 | ~26 s |
| 358-789 | 4 | 11 min measured at `-j6` under load average 62-87 |

Red-before and green-after, on real trees that pass every existing gate:

| Range | Head | Result |
|---|---|---|
| `76fefdca7..77cedc9a5` | `5263ac31f^` | **RED**, 1 of 40 units: `ltx2_connector_gemm_probe.cpp:41:1: error: multi-line comment [-Werror=comment]`. 25.1 s |
| `76fefdca7..5263ac31f` | the fix | GREEN, 40 of 40 units, 25.5 s |
| `a7e08aae1..6d803da46` | `6d803da46` | **RED**, 2 of 58 units, 101.8 s |
| the same range, both upstream repairs applied to the worktree | | GREEN, 58 of 58 units, 108.9 s |
| `ee5c86031..c27246d37` | `08fa2f5aa^` | **RED**, 1 of 789 units, 663.9 s at `-j6` under load average 62-87 |

**The `6d803da46` row is the one that decides the design.** That commit is where
both of the GLM defects entered, and the gate names both of them at once:

```
FAILED to compile src/vllm/model_executor/models/glm_moe_dsa_forward.cpp
  glm_moe_dsa_forward.cpp:456:73: error: cannot convert
    'vllm::mla::MlaSharedSelection*' to 'vt::Tensor*'
FAILED to compile tests/vllm/models/test_glm_moe_dsa_schedule.cpp
  test_glm_moe_dsa_schedule.cpp:304:86: error: cannot convert
    'vllm::mla::MlaSharedSelection*' to 'vt::Tensor*'
```

The first is what `11f34effb` fixed hours later; the second is `08fa2f5aa`, which
`11f34effb`'s own message says it missed. **Neither file has to be in the diff.**
`git diff --name-only 6d803da46^ 6d803da46` lists 18 paths and does not name
`test_glm_moe_dsa_schedule.cpp`; the unit enters the affected set only through
`include/vllm/model_executor/models/mla_attention.h`, by the reverse-include
closure. A changed-`.cpp` scope is green on that tree.

The last row is the same defect found from a whole-campaign range -- 227 changed
paths, 789 units -- and it is the worst-case cost figure: 11 minutes at `-j6` on
a box whose load average was between 62 and 87 throughout. That run also prints,
by name, the seven units it did NOT check because no target in this
configuration compiles them: five `.cu` files and two Tenstorrent units. Limit 2
below is therefore visible in the output of every run, not only in this document.

Mutations, each applied to a scratch copy and restored under a sha256 check:
delete the reverse-include closure and case 2 must fail; turn the exit-2 arm
into exit 0 and cases 4 and 5 must fail; drop `-Werror` from the reconstructed
command and case 1 must fail; fold the uncompiled sources into the compiled
total and the un-run case must fail; drop the forced inclusion and case 8 must
fail; delete preflight's call site and case 7 must fail.

## Risks

- **The checker is itself a TU consumer.** If `compile_commands.json` goes stale
  against a moved source tree, the checker compiles the wrong file. Mitigated by
  configuring fresh into a scratch directory on every run; 1.7 s buys that.
- **Parallelism on a shared box.** `-fsyntax-only` allocates far less than a
  codegen-and-link job, and jobs default to half the CPU count capped at 8.
  Parallel agent builds have OOM-killed this box before.
- **A wide header change costs minutes, on EVERY preflight run.** `3bfd1a738`
  touched a core `vt` header and reaches 783 TUs; a whole-campaign range reaches
  789 and took 4.1 min measured. That is the correct answer for that change and
  it is still cheaper than the build it replaces, but preflight runs more than
  once per branch and pays it again each time. No cap is imposed, because a cap
  below the real fanout is a mute switch. The bounded fix is a result cache
  keyed on each unit's dependency-set hash and its recorded command, which would
  reduce a repeat run to the dependency scan; it is named under `## Owed` and is
  not built here.

## Gates

- `python3 tests/scripts/test_check_tree_compiles.py`
- `python3 scripts/check-tree-compiles.py --base origin/main`
- `scripts/agent-preflight.sh`

## Evidence

Measured on this box, 20 cores, with concurrent agent builds running throughout:
load average ~28 for the microbenchmarks and ~62 for the two historical runs.
Every figure is therefore pessimistic, and the two that matter most were taken
under the worse of the two.

| Quantity | Value |
|---|---|
| `cmake` configure, Ninja, Release | 1.67 s, 14 MB |
| TUs in `compile_commands.json` | 1218 |
| `c++ -MM -MG` per TU | 0.02-0.10 s |
| whole-tree dependency scan at `-j8` | 15.9 s |
| `c++ -fsyntax-only` per TU | 0.27-3.26 s, mean 1.76 s over 10 sampled TUs |
| full `cmake --build`, warm | ~12 min, 9.4 GiB |

**The two costs the row is judged on**, both measured end to end with the gate
as it ships:

| Change shape | Command | Cost |
|---|---|---|
| records-only (this branch: 8 paths, no C++) | `check-tree-compiles.py --base origin/main` | **0.20 s**, no configure, no compiler |
| a code change (a row's worth of `.cpp` and `.h`: 40 TUs) | `--base 76fefdca7 --head 5263ac31f` | **25.5 s** (1.4 s configure + 12.7 s scan + 11.3 s compile) |

Affected-TU fanout over the 60 commits ending at `9fa3be388`, which is what
makes the first number the common one:

| TUs affected | Commits | Gate cost |
|---:|---:|---|
| 0 | 30 | ~0.2 s, no configure |
| 1-4 | 20 | 3-22 s |
| 18-37 | 6 | ~26 s |
| 358-789 | 4 | minutes; see the row below |

Red-before and green-after, on two real trees that pass every existing gate:

| Range | Head | Result |
|---|---|---|
| `76fefdca7..77cedc9a5` | `5263ac31f^` | **RED**, `tools/bench/ltx2_connector_gemm_probe.cpp:41:1: error: multi-line comment [-Werror=comment]`, 1 of 40 units, 25.1 s |
| `76fefdca7..5263ac31f` | the fix | GREEN, 40 of 40 units, 25.5 s |
| `ee5c86031..c27246d37` | `08fa2f5aa^` | **RUNNING at this commit.** Scope resolved and the closure computed: 227 changed paths, 789 units to check, 7 named as uncompiled. The verdict is not in yet and is NOT claimed here; the follow-up commit records it |
| `ee5c86031..08fa2f5aa` | the fix | queued behind the row above, same range, same worktree |

**The second range is the one that decides the design.** Its base is
`ee5c86031`, the commit that ADDED the failing test, so
`test_glm_moe_dsa_schedule.cpp` is not among its 227 changed paths -- verified,
`git diff --name-only ee5c86031 c27246d37` does not name it. The unit enters the
affected set only through `include/vllm/model_executor/models/mla_attention.h`,
by the reverse-include closure. A changed-`.cpp` scope is green on that tree.

The same run also prints, by name, the seven units it did NOT check because no
target in this configuration compiles them: five `.cu` files and two Tenstorrent
units. Limit 2 below is therefore visible in the output of every run, not only
in this document.

Mutations, each applied to a scratch copy of the checker and restored
byte-for-byte with a sha256 check, against a green 15-case baseline:

| Mutation | Cases that turned red |
|---|---|
| delete the reverse-include closure | `test_a_changed_header_reaches_a_translation_unit_the_diff_does_not_name` |
| `CANNOT_VERIFY = 2` becomes `0` | the two `..._is_cannot_verify_...` cases |
| drop `-Werror` from the recorded command | the two `-Wcomment` cases |
| fold the uncompiled sources into the compiled total | `test_a_source_no_target_compiles_is_named_and_not_counted` |
| drop the forced inclusion of unscannable units | `test_a_unit_whose_dependency_scan_fails_is_checked_anyway` |
| delete preflight's call site (reachability) | the two preflight-mapping cases and the registration case |

Each mutation reddens the cases that name its guarantee and no others, and the
baseline is green before and after every one.

## What this does NOT cover

Stated here because a checker's message defines what it enforces, and no gate
checks that this prose and that message agree.

1. **It does not link.** An undefined symbol, a duplicate definition, a missing
   vtable, an ODR violation and an unregistered CTest target all pass.
2. **One configuration.** The default Linux host `c++`, `Release`, with CUDA,
   HIP, Metal and MSVC off. A `.cu`, `.hip` or `.mm` TU is not in
   `compile_commands.json` and is not compiled. `444` — ROCm `main` does not
   build — is exactly the class this cannot see.
3. **One compiler.** A `clang`-only or `gcc-15`-only diagnostic stays CI's.
4. **It runs nothing.** A tree that compiles can fail every test in it.
5. **Include edges only.** A TU a change reaches through a CMake option, a
   generated header written by a script the diff changed, or an embedded data
   file is not pulled into scope.
6. **Build files partially.** A `CMakeLists.txt` edit is covered for configure
   errors and for TUs the diff names. An edit that changes flags for an existing
   TU it does not name is not re-verified.
7. **The tree at HEAD, not each commit.** A range red at commit 3 and green at
   commit 5 reads green, which is the right answer for what is about to be
   pushed and the wrong one for a bisect.

## Owed

- A base that moves forward past a branch's own commits narrows the committed
  component of the scope. Every range gate in preflight carries this; it is not
  introduced here and it is not closed here.
- Limit 6: comparing `compile_commands.json` at base and at head would cover a
  flag-only build edit exactly, at the cost of a second configure and a base
  worktree. Not built; #2401 owns it.
- Limit 2: the non-default configurations stay CI-only. No row claims moving
  them earlier. The checker names the units it could not check for this reason
  in its own output -- on the `08fa2f5aa^` range that is five `.cu` files and
  two Tenstorrent units -- so the uncovered set is visible per run rather than
  only in this document.
- A repeat preflight run on an unchanged wide branch recompiles the same units.
  A cache keyed on `(unit, hash of its dependency set, recorded command)` would
  reduce it to the dependency scan. Not built; #2401 owns it.

## Stop conditions

Return `NEEDS_DECISION` rather than widening what preflight demands of a
`read-only` session, or adding a cap that suppresses a wide fanout.

## Now

Landed: `scripts/check-tree-compiles.py`, its 15-case suite, and its wiring into
`scripts/agent-preflight.sh` and the CI script lane. The gate reddens on
`5263ac31f^` and on `6d803da46`, where both GLM defects entered, and greens on
their fixes. A records-only change costs 0.20 s and a row-sized code change
25.5 s.

Owed and unclaimed: the repeat-run cache, the base-vs-head `compile_commands`
comparison for a flag-only build edit, and the non-default configurations, all
listed under `## Owed` and tracked by #2401. No `AGENTS.md` edit was made; the
argument for leaving the policy prose alone is in `## Design`, and stating the
obligation in prose as well remains the developer's call.
