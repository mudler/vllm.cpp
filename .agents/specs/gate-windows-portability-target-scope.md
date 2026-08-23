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

## 10. The index row this row now owns alone

`.agents/issue-index.md` carried [#1649](https://github.com/mudler/vllm.cpp/issues/1649)
TWICE on `main` at `038ff61e5`, so `scripts/check-agent-record.py` exited 1 for
every session and every pull request cut from it:

```
ERROR: .agents/issue-index.md: issue #1649 listed twice. Under `merge=union` a
duplicate is what two branches appending the same issue look like
```

`:592` was appended by `a7bb3130b` (PR [#1633](https://github.com/mudler/vllm.cpp/pull/1633))
under `ENG-HF-MODEL-DOWNLOAD`, the lane that FILED the issue. `:632` was
appended by `2f2a70925` (PR [#1701](https://github.com/mudler/vllm.cpp/pull/1701))
under this row, the lane that FIXED it. Neither lane could see the other's
append, and `merge=union` combines two appends silently.

`:592` was dropped and this row's `:632` kept. The test is the one `AGENTS.md`
states under `## Every change starts from an issue`: the issue is linked "in
three places that must agree: the index, the row's spec, and the pull request
body". Only one of the two rows passes it. This spec names `Issue: #1649` on
its third line and PR #1701 names #1649 in its title, so `:632`'s ownership
claim is corroborated on all three surfaces. `.agents/specs/hf-model-download.md`
mentions #1649 nowhere at all — it is absent from that spec's `## Owed` and from
every other section — so `:592`'s ownership claim was never corroborated by the
spec it pointed at. Keeping `:592` would have left the index asserting an
ownership the tree does not support; keeping `:632` leaves the assertion true.

`:592` carried four facts `:632` does not, and they are recorded here so that
dropping the row costs the reader nothing:

1. **Where the `/w` came from.** `a50c57d69`
   (PR [#1505](https://github.com/mudler/vllm.cpp/pull/1505), row
   `ENG-HF-MODEL-DOWNLOAD`, [#1280](https://github.com/mudler/vllm.cpp/issues/1280))
   added the static-BoringSSL transport, and with it the
   `$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>` generator expression that §2 quotes.
   `git log -S` on that expression returns that commit alone.
2. **The run that measured the red.** `9f13751c3`, PR #1633's head, in job
   [96949585684](https://github.com/mudler/vllm.cpp/actions/runs/32540549699/job/96949585684).
   The refusal lands seventeen seconds after `Build files have been written to`,
   and the job log carries no `error C####` and no `error LNK####`, which is how
   the filing established that no translation unit is ever read.
3. **What the red is NOT.** Not
   [#503](https://github.com/mudler/vllm.cpp/issues/503), a baseline-reporting
   hole; not [#603](https://github.com/mudler/vllm.cpp/issues/603), a POSIX
   `setenv` in `test_backend_cross_device.cpp`; and not
   [#965](https://github.com/mudler/vllm.cpp/issues/965), a C4456 shadow in
   `server_main.cpp`. None of the three is a checker refusal, and none names
   this string. §8 already sends the compile itself back to those three; this
   is the separation that put the refusal outside them.
4. **How it was isolated, and by whom.** Dropping only the MSVC arm of that
   generator expression made the same checker print `Windows portability
   contract OK`, and the tree was restored byte-for-byte against a pre-taken
   sha256. Found while repairing the fresh-review findings on PR #1633, which
   reads the red, is not its cause, and does not touch `CMakeLists.txt`.

The dropped row's verbatim text is in the body of the commit that dropped it.
`git log -S'#1649' -- .agents/issue-index.md` finds that commit, and
`git show <sha>` prints both the argument and the row.
