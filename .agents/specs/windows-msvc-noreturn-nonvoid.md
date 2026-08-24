# Native MSVC `[[noreturn]]` on a non-void return type — C4646/C2220

Identity: `ENG-RELEASE-WINDOWS` (checker), `MODEL-MM-dots3-note` (the defect site)

Issue: [#1829](https://github.com/mudler/vllm.cpp/issues/1829)

Parent specification:
[windows-binary-release.md](windows-binary-release.md)

Predecessor repairs of the same shape — a POSIX-only or non-standard C++ spelling
that only MSVC rejects, found on a pull-request-only Windows lane:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md),
[windows-msvc-m-pi.md](windows-msvc-m-pi.md).

Status: `ACTIVE`. Base `404f0cdc5b876cbd8dec9d9cb97691ce73ec63cf`.

## 1. Scope

Two changes, and the second is what stops the first from recurring.

1. Remove `[[noreturn]]` from the `Dots3NoteModel::ForwardDevice` declaration in
   `src/vllm/model_executor/models/dots3_note.h`, and correct the comment above
   it, which is wrong on two independent counts (§3).
2. Teach `scripts/check-windows-portability.py` to refuse `[[noreturn]]` applied
   to a function whose return type is not `void`, sweeping the whole first-party
   C-family tree rather than only the shipped-server closure, and wire the
   checker and its suite onto a lane that runs.

Explicitly excluded:

* The refusal itself. `VT_CHECK(false, ...)` in `dots3_note.cpp` is untouched:
  it is what earns the model its `REFUSE` classification (§3b), and the row that
  owns the language tower (#699) owes the maths, not this repair.
* Any restructuring of `Dots3NoteModel`, its signature, or its message.
* `src/vllm/model_executor/models/dots3_note.cpp`. The definition carries no
  attribute; only the declaration does.
* `third_party/`. Vendored code is compiled with `/w` and this project does not
  author it.
* [#584](https://github.com/mudler/vllm.cpp/issues/584), the Windows RUNTIME
  crash. It is a different failure that the current build break MASKS — a binary
  that does not link cannot crash — and unmasking it is its progress, not its
  regression.

## 2. Observed baseline

`windows-msvc-cpu` run 32684932825, job 97308185894, `main` at `4f0d44ca2`:

```
src\vllm\model_executor\models\dots3_note.cpp(606,31): warning C4646: function declared with 'noreturn' has non-void return type
src\vllm\model_executor\models\dots3_note.cpp(606,31): error C2220: the following warning is treated as an error
```

`cmake --build` then exits 1 and the whole `vllm` project fails to compile.
`build-test-cpu`, `build-test-cpu-arm64`, `build-newest-gcc` and `verify (cpu)`
are green on the same commit: GCC and Clang accept the declaration silently.

The attribute is on the DECLARATION, not the definition. MSVC reports the
definition's line because that is where it reconciles the two.

Introduced by `849a7dd73` (#1805, row `MODEL-MM-dots3-note`, lane #699).

## 3. Root cause, and two false premises recorded beside it

`src/vllm/model_executor/models/dots3_note.h` declares

```cpp
  [[noreturn]] static ForwardLogits ForwardDevice(
```

`ForwardLogits` is not `void`, so `[[noreturn]]` is C4646 under MSVC and this
build is `/W4 /WX`.

The comment above the declaration argues for the attribute, and both of its
arguments are false. Correcting the code without correcting the comment would
leave the next reader the same two reasons to put it back.

(a) **"The shape mirrors `KimiK3Model::ForwardDevice`."** It does not.
`include/vllm/model_executor/models/kimi_k3.h` declares
`static ForwardLogits ForwardDevice(` with NO attribute. dots3 did not mirror
the tree's other refuse-by-name model; it ADDED something that model never had.
Removing the attribute is what makes the sentence true.

(b) **"So `scripts/check-runner-routing-consistency.py` classifies it REFUSE
instead of dropping it into the silently-exempt NONE bucket."** That checker
never reads the declaration. `_REFUSE = re.compile(r"VT_CHECK\(\s*false")` is
matched against the function BODY by `classify_body`. The `VT_CHECK(false, ...)`
in `dots3_note.cpp` is the whole of what earns `REFUSE`, and it is untouched
here. Measured both sides (§6).

## 4. Design

### 4.1 The fix

Delete `[[noreturn]] ` from the declaration. Rewrite the comment to state what
is true: the signature matches `KimiK3Model::ForwardDevice`, the body's
`VT_CHECK(false, ...)` is what the routing checker classifies on, and
`[[noreturn]]` is not available on this signature because MSVC rejects it on a
non-void return type.

`ForwardDevice` cannot simply become `void`: it is the shape W3 fills in and the
shape the registry hook delegates to. Dropping the attribute is therefore the
minimum complete change, and it costs nothing — the attribute is an optimisation
and diagnostic hint, never a correctness requirement, and the function's only
statement is a `VT_CHECK(false, ...)` that throws.

### 4.2 The detector

`noreturn_nonvoid_sites(source)` returns `(line, return type)` for every
`[[noreturn]]` whose function's declared return type is not `void`.

Method, and why each step is there:

* Blank comments and string/char/raw literals with
  `without_cpp_comments_and_literals`, which preserves offsets and newlines, so
  the reported line number is the source line. This is what keeps
  `dots3_note_registry.cpp`'s prose `// [[noreturn]] until W3-W10 land` out of
  the population.
* From the end of the attribute, scan forward to the declarator's opening `(`,
  tracking `<`/`>` so a template argument list does not end the scan early.
  A `;`, `{` or `}` reached first means this is not a function declaration, and
  the site is reported as unparsed rather than silently dropped — the checker
  fails closed, and a shape it cannot read is a shape a human should look at.
* From the text between, drop further `[[...]]` attributes, `extern "C"`, and
  the decl-specifiers `static inline constexpr consteval extern virtual explicit
  friend mutable thread_local`. `*` and `&` are split out as their own tokens so
  `void*` cannot read as `void`.
* The last remaining token is the declarator-id; everything before it is the
  return type. No tokens at all means a constructor or destructor, which has no
  return type and is accepted.
* `auto` is accepted only when a trailing `-> void` follows the parameter list.

Deliberately NOT done: no allowlist of export or attribute macros. An unknown
all-caps token stays part of the return type and reds the gate. That direction
is the fail-closed one, matching this checker's stated polarity, and the tree
has no such macro on a `[[noreturn]]` today.

Known limit, stated rather than hidden: only the LEADING attribute position is
read. `void f [[noreturn]] ();` — legal, and applying to the declarator — is not
detected. No site in the tree uses it.

### 4.3 Scan set, and why it is not the shipped-server closure

Every other rule in this checker reads `shipped_server_sources(...)`, the
closure reachable from the shipped server target. C4646 does not respect that
boundary: `windows-msvc-cpu` builds the tests too, so a test translation unit
carrying the attribute reds the same lane.

Widening the POSIX rules to `tests/` is a measurement problem — a guarded POSIX
call and an unguarded one look alike ([#1107](https://github.com/mudler/vllm.cpp/issues/1107)).
This rule has no such ambiguity: `[[noreturn]]` on a non-void return type is
unconditionally ill-formed under MSVC wherever it appears. So the new rule
sweeps the tracked first-party C-family tree by glob — `src include tests
examples tools benchmarks scripts`, suffixes `.c .cc .cpp .cxx .h .hh .hpp .hxx
.cu .cuh .hip .inc`, minus `third_party/` — and needs neither CMake nor a build
directory. The narrowing of the other rules is untouched.

`.cu`/`.cuh` are in because nvcc hands host code to `cl.exe` on Windows.

### 4.4 Wiring

A checker no job runs catches nothing, which is the whole of
[#680](https://github.com/mudler/vllm.cpp/issues/680) and half of
[#646](https://github.com/mudler/vllm.cpp/issues/646). Measured on this base:
`grep -rn 'windows-portability' .github/` returns nothing, and
`test_check_windows_portability` is absent from `scripts/agent-preflight.sh`'s
`SUITES` array. The other half of those issues — a `main` baseline for the
Windows lanes themselves — is not in this scope and stays open.

* `.github/workflows/ci.yml`, job `agent-record`: a step running
  `scripts/check-windows-portability.py` and its full suite. That job carries no
  `if:` — which `check-test-registration.py` requires before it counts a step as
  a registration — and it already runs every other tree checker.
  `shipped_server_sources` forces `-G Ninja`, and `ninja-build` is not assumed
  present on `ubuntu-latest` (every other job in this file installs it
  explicitly), so the step installs it and prints both tool versions first: a
  missing tool must be legible, not a mystery configure failure.
  Measured cost on this base: 6.3 s for the checker, 18.1 s for the suite.
* `scripts/agent-preflight.sh`: `test_check_windows_portability` added, guarded
  on `cmake` and `ninja` being on `PATH` and SKIPPED with a reason when they are
  not. This follows the `test_ltx25_render_compare` precedent already in that
  file: a missing tool is a skip and never an `ok`, and the lane that must not be
  silent is CI, which installs the tool.

## 5. Tests

`tests/scripts/test_check_windows_portability.py`, class
`NoReturnNonVoidTests`:

* RED-FIRST, and it is the whole point: `test_real_tree_has_no_noreturn_on_a_non_void_return`
  runs the detector over the REAL tree and asserts zero sites. It fails on this
  base naming `dots3_note.h:282`, and passes after the fix. This is the case that
  would have caught #1829 on Linux.
* Non-vacuity: `test_real_tree_scan_reaches_the_noreturn_population` asserts the
  sweep visits at least 40 files that contain the token, so the zero above cannot
  be a zero because nothing was read. A floor below the real count is a mute
  switch.
* Positive cases: plain non-void, `void*`, a template return type, `auto` with a
  non-void trailing return, and the exact `dots3_note.h` shape across two lines.
* Negative cases: `void`, `static void`, `extern "C" ... void`, a const member
  function, `auto ... -> void`, a destructor, the token inside a `//` comment,
  and the token inside a string literal.
* `test_check_reports_a_noreturn_non_void_site` drives the full `check()` on a
  fixture tree and asserts the error string, so the rule is reached through the
  checker's own entry point and not only through the helper.
* `WindowsPortabilityWiringTests` asserts the wiring itself:
  `test_ci_runs_the_checker_and_its_suite_unconditionally` reads `ci.yml` with
  `yaml.safe_load`, requires the `agent-record` job to carry no `if:`, and
  requires the `ninja-build` install to precede the checker call in the same
  step body; `test_preflight_runs_the_suite_or_skips_with_a_reason` requires
  `agent-preflight.sh` to carry the `run` line, the `skip` line and the module
  name exactly once each. A detector on a lane nobody runs is the defect this
  change exists to avoid, so the wiring is asserted, not assumed.

## 6. Gates

```sh
python3 -m unittest tests.scripts.test_check_windows_portability
python3 scripts/check-windows-portability.py
python3 scripts/check-runner-routing-consistency.py
python3 tests/scripts/test_check_runner_routing_consistency.py
python3 scripts/check-symbol-anchors.py
scripts/agent-preflight.sh --fail-on-skip
```

The routing checker runs on BOTH sides of the fix, because §3b claims the
classification does not move and a claim about a classifier is measured, not
reasoned about.

## 7. Risks

| Risk | Handling |
|---|---|
| The detector's parser is a heuristic over C++ declarations, and a false positive would red an unrelated pull request. | The population is small and enumerated: 53 files carry the token, 57 sites, and the detector flags exactly one. Fifteen unit cases pin the accept/reject boundary in both directions. A shape it cannot parse is reported rather than dropped, so the failure direction is a visible red, not a silent hole. |
| Widening the scan set to `tests/` and `examples/` widens what can red a lane. | Only for this rule. The POSIX and path-conversion rules keep the shipped-server closure. The new rule's predicate is unconditional ill-formedness, not a guarded/unguarded judgement. |
| The `agent-record` step adds an apt install and a CMake configure to a widely-depended job. | Both tool versions print first so a missing tool is legible. The configure runs with every backend OFF and measured 6.3 s. |
| Removing `[[noreturn]]` loses a diagnostic hint. | The function's only statement is a throwing `VT_CHECK(false, ...)`; no caller relies on non-return for flow analysis, and the attribute was never on the model this one claims to mirror. |
| The comment correction could drift again once W3 lands. | The corrected comment states the mechanism (the body's `VT_CHECK(false, ...)`) rather than a claim about another file's declaration, so it stays true when W3 replaces the body. |

## 8. Stop conditions

* Stop and report `NEEDS_DECISION` if removing the attribute changes the routing
  checker's classification of `dots3_note` away from `REFUSE`.
* Stop and report if the sweep finds a second site whose repair is not the same
  one-token deletion — that is a different semantic decision and a different row.
* No MSVC host is available here, so the Windows compile is verified by the
  `windows-msvc-cpu` and `windows-msvc-vulkan` jobs on this pull request. Their
  verdict is the closing evidence for the compile; the detector is the closing
  evidence for the class.

## 9. Evidence

Base `404f0cdc5b876cbd8dec9d9cb97691ce73ec63cf`, pull request
[#1840](https://github.com/mudler/vllm.cpp/pull/1840).

RED, with the detector in place and the header unfixed:

```
$ python3 scripts/check-windows-portability.py; echo "RC=$?"
ERROR: src/vllm/model_executor/models/dots3_note.h:282: [[noreturn]] on a non-void return type 'ForwardLogits' is MSVC C4646, and /W4 /WX makes it C2220
RC=1
```

`NoReturnNonVoidTests` + `WindowsPortabilityWiringTests`: 12 tests, 2 failures —
`test_real_tree_has_no_noreturn_on_a_non_void_return` naming
`dots3_note.h:282`, and `test_real_tree_declaration_still_carries_no_attribute`
naming `  [[noreturn]] static ForwardLogits ForwardDevice(`.

GREEN, after removing the attribute:

```
$ python3 -m unittest tests.scripts.test_check_windows_portability
Ran 94 tests in 24.339s
OK
$ python3 scripts/check-windows-portability.py; echo "RC=$?"
Windows portability contract OK
RC=0
```

Sweep: 57 `[[noreturn]]` sites across 53 token-carrying files; one violation, no
siblings.

Routing classification, measured on both sides through
`scan_registrations(MODELS_DIR, INCLUDE_DIR)`: all 33 registered models identical
byte-for-byte, `dots3_note` `REFUSE` / `BF16_RESIDENT` before and after, and both
`OK (runner-routing)` / `OK (bf16-activation)` lines unchanged. §3b holds.

Compile: `dots3_note.cpp.o` and `dots3_note_registry.cpp.o` build clean under
GCC 13 (`ninja` exit 0). No MSVC host was available, so the C4646 side is
verified by this pull request's `windows-msvc-cpu` and `windows-msvc-vulkan`
jobs.

Mutation of the wiring: deleting the `check-windows-portability.py` line from
`ci.yml` and the `run` line from `agent-preflight.sh` reds both
`WindowsPortabilityWiringTests` cases (2 of 2). Tree restored.

`scripts/agent-preflight.sh --fail-on-skip`: `All gates green.`, RC 0, no FAIL
and no SKIP. `test_check_windows_portability` reports `ok` in that run, which is
the new preflight registration executing.

## Outcome

Filled in at `DONE`.
