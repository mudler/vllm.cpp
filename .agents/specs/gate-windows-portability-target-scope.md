# Spec — the MSVC warning policy is asserted by TARGET, not by file

Issue: [#1649](https://github.com/mudler/vllm.cpp/issues/1649)
Row: `GATE-WINDOWS-PORTABILITY-TARGET-SCOPE` (unplaced record/gate defect; the
tracked tree is a checker, not a matrix row)
State: `ACTIVE`

## 1. Scope

`scripts/check-windows-portability.py` refuses a tree whose MSVC `/W4 /WX`
policy is negated on the C/C++ compile. It read `/w` anywhere in
`CMakeLists.txt` as that negation. This spec narrows the read to the flags that
reach a target THIS project declares, and pins that the narrowing admits exactly
that and nothing else.

Out of scope: every other rule in the checker, and the `/W4 /WX` policy itself.

## 2. The defect, measured

`windows-msvc-cpu` and `windows-msvc-vulkan` were red on `main` and on every
pull request. It is not a compile failure:
`scripts/build-windows-release.ps1:31` runs the checker before anything is
compiled, and it exits 1 with

```
ERROR: CMakeLists.txt: MSVC /W4 /WX policy is negated on the C/C++ compile by /w
```

`CMakeLists.txt:2436-2443` is the whole cause:

```cmake
      # Vendored code is not on this project's -Werror path.
      foreach(_boringssl_target ssl crypto)
        if(TARGET ${_boringssl_target})
          target_compile_options(${_boringssl_target} PRIVATE
            $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
        endif()
      endforeach()
```

The same defect reds this checker's OWN suite, which #1649 did not record:
`tests/scripts/test_check_windows_portability.py`
`test_real_tree_msvc_warning_policy_reaches_the_cxx_compile` fails with
`negation='/w'` on `main` at `8540a2755` (78 tests, 1 failure). So the tree
carried the contradiction in two places at once.

The checker had no scope model at all. `check()` set `warnings = cmake` — the
entire file — then concatenated `cmake/CompilerWarnings.cmake` and did a flat
token search over both. Under that reading a global `add_compile_options(/w)`
and a `target_compile_options(<vendored> PRIVATE /w)` are indistinguishable,
though only the first negates the policy.

## 3. Design

`without_foreign_target_compile_options(text)` blanks each balanced
`target_compile_options(...)` span whose target provably names only targets this
project never declares. Blanking rather than cutting keeps reported offsets
meaningful, matching `without_set_source_properties` and `msvc_cxx_flag_text`.

A target is "foreign" only when it is PROVABLY so:

* a literal name absent from `project_targets()`, which collects every
  `add_library` / `add_executable` first argument;
* a `${VAR}` whose `foreach(VAR ...)` binding resolves to names that are all
  absent from that set.

Everything else stays in scope. The fail-safe direction is deliberate: an
unbound `${...}` could name a project target, so it still answers for the
policy.

`cmake/CompilerWarnings.cmake` is kept WHOLE. It is the policy module, and it
applies the flags through a function parameter (`target_compile_options(${target}
PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)`) that no caller-independent reading
can resolve. Scoping it would blank the policy itself and turn the "required"
arm red.

## 4. Risks / decisions

The narrowing is a widening of what the checker ACCEPTS, so the risk is a hole.
It is bounded by three properties, each pinned by its own test and each proved
discriminating by mutation:

* a project target may not silence the policy (`vllm` is declared by
  `add_library`, so `target_compile_options(vllm PRIVATE /w)` stays refused);
* a `foreach` binding naming a project target is not laundered by the vendored
  construct;
* an unresolvable target stays in scope.

Residual hole, stated rather than hidden: a negation on a project target reached
through a variable this checker cannot resolve — a `set()` binding rather than a
`foreach()` one — is not caught. Closing it means evaluating `set()` bindings,
which is a larger reading of CMake than any rule here currently performs. No
such construct exists in the tree today.

The alternative — dropping the MSVC arm of the generator expression — was
rejected: it puts a vendored dependency's diagnostics on this repository's
`-Werror` path, which is exactly what the comment above it exists to prevent.

## 5. Tests

`tests/scripts/test_check_windows_portability.py`, four added cases plus one
rescoped:

| Case | Before | After |
|---|---|---|
| `test_a_vendored_target_may_silence_its_own_warnings` | RED | green |
| `test_real_tree_msvc_warning_policy_reaches_the_cxx_compile` | RED (`negation='/w'`) | green |
| `test_a_project_target_may_not_silence_the_policy` | green | green |
| `test_a_foreach_binding_naming_a_project_target_is_not_vendored` | green | green |
| `test_an_unresolvable_target_stays_in_scope` | green | green |

The three that were green before are the guard properties. They are proved
discriminating rather than asserted: forcing `_target_is_foreign` to `return
True` (maximal widening) reds exactly those three and nothing else (82 tests, 3
failures), and the tree was restored byte-for-byte after the probe.

## 6. Gates

* `python3 -m unittest tests.scripts.test_check_windows_portability` — 82 tests,
  `OK`. Was 78 tests / 1 failure on `main`.
* `python3 scripts/check-windows-portability.py` on the real tree —
  `Windows portability contract OK`, rc 0. Was rc 1 on `main`.

The Windows CI legs themselves are not runnable here; the checker they run
first is, and it is the step that refused.

## 7. Stop conditions

Stop and ask if closing the residual `set()`-binding hole is wanted in this
change: it is a materially larger CMake reading and belongs to its own row.

## 8. Now

The narrowing has landed with its tests. `windows-msvc-cpu` and
`windows-msvc-vulkan` should clear their pre-compile refusal; whether they then
pass the compile itself is unproven here and is #503/#603/#965 territory.

## 9. Outcome

Recorded on landing.
