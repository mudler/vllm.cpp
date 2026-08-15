# MSVC `/W4 /WX` policy — assert the TOKEN on the C/C++ compile, not a substring anywhere in the file

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#774](https://github.com/mudler/vllm.cpp/issues/774)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Related repair: [windows-msvc-strict-build.md](windows-msvc-strict-build.md),
which established the strict warning policy this checker line exists to hold.

Status: `ACTIVE`. Base `af026e5241e16e1d2a89da001b978ffcb3e0f634`.

## Scope

Replace the substring assertion at `scripts/check-windows-portability.py:1710`
with a token-boundary assertion evaluated over the flags that actually reach an
MSVC **C/C++** compile.

In scope:

1. Token boundaries for `/W4` and `/WX`, so a longer flag that merely *contains*
   the token no longer satisfies the policy.
2. Language scoping, so an occurrence confined to a non-C/C++ `COMPILE_LANGUAGE`
   generator expression (OBJCXX, CUDA, ...) cannot satisfy the policy.
3. Comment stripping, so the policy cannot be satisfied by prose.
4. Rejecting the *negating spellings of the two tokens the policy names* when
   they reach the C/C++ compile: `/WX-`, `/W0`, `/w`.

Explicitly excluded, argued in **Sibling evasions** below: the
`COMPILE_WARNING_AS_ERROR` target/cache property, and per-warning `/wd####`
suppressions.

Nothing outside `scripts/check-windows-portability.py`,
`tests/scripts/test_check_windows_portability.py`, this spec and the roadmap
issue table changes. `cmake/CompilerWarnings.cmake` is **not** edited: the tree
already satisfies the repaired policy, and a checker repair that also moves the
thing it checks cannot show that it holds.

## Observed baseline and root cause

`scripts/check-windows-portability.py:1710` at base:

```python
if not all(token in warnings for token in ("/W4", "/WX")):
    errors.append("CMakeLists.txt: MSVC /W4 /WX policy is required")
```

`warnings` is the concatenation of `CMakeLists.txt` and
`cmake/CompilerWarnings.cmake`, unparsed. Three independent blindnesses follow
from `in`:

- **Inversion.** `"/WX" in "/WX-"` is `True`. `/WX-` is MSVC's spelling for
  *disable* warnings-as-errors — the exact inversion of the policy.
  `"/W4" in "/W44996"` is `True` too, and `/W44996` sets warning C4996 to level
  4 rather than raising the warning level.
- **Language.** The test asks only whether the bytes occur *somewhere*. A `/WX`
  reachable only through `$<$<COMPILE_LANGUAGE:OBJCXX>:...>` is inert under
  MSVC — Objective-C++ is the Metal backend and never compiles on Windows — yet
  satisfies the policy.
- **Prose.** `CMakeLists.txt:30` contains the literal text `/W4 /WX` inside a
  `#` comment. That comment alone satisfies the base assertion, so the policy
  survives deleting every real flag in the tree.

The measured demonstration is PR
[#640](https://github.com/mudler/vllm.cpp/pull/640) commit `74ba3823f`, whose
`cmake/CompilerWarnings.cmake` used all three at once (reverted in `15aa963a6`
after review):

```cmake
  if(MSVC)
    set_property(TARGET ${target} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/WX->
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4458>
      $<$<COMPILE_LANGUAGE:OBJCXX>:/W4>
      $<$<COMPILE_LANGUAGE:OBJCXX>:/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
  endif()
```

The only bare `/WX` left is on OBJCXX. The gate passes.

## Design

Two helpers next to the existing CMake-text helpers
(`without_set_source_properties`, `source_properties`), then a rewritten
assertion.

`msvc_cxx_flag_text(text)` returns the flag text that can reach an MSVC C/C++
translation unit:

- `#` comments removed, using the same `re.sub(r"(?m)#.*$", "", ...)` idiom
  `_has_central_msvc_nominmax` already uses.
- Every generator expression whose `COMPILE_LANGUAGE` mentions name only
  languages outside `{C, CXX}` blanked. Spans come from a `$<` / `>` stack scan,
  so `$<$<COMPILE_LANGUAGE:OBJCXX>:/WX>` is blanked whole, `$<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>`
  is kept whole, and a genex naming no language at all (`$<$<CONFIG:Debug>:/WX>`)
  is kept, because it does reach C/C++. Blanking preserves offsets, matching
  `without_set_source_properties`.

`has_msvc_flag(text, flag)` matches the flag with boundaries on both sides:
`(?<![A-Za-z0-9_-])` before and `(?![A-Za-z0-9_-])` after. `/WX-` therefore does
not answer for `/WX`, `/W44996` does not answer for `/W4`, and `/wd4324` does
not answer for `/w`. Matching stays case-sensitive because `cl` is: `/w` and
`/W4` are different flags, not two spellings of one.

The assertion then reads: both `/W4` and `/WX` must be present in that text, and
none of `/WX-`, `/W0`, `/w` may be. The first error message keeps its existing
prefix (`CMakeLists.txt: MSVC /W4 /WX policy is required`) so nothing that greps
for it breaks, and names which token is missing; the negation is a second,
separately-worded error, because "the flag is absent" and "the flag is present
and cancelled" are different repairs.

Rejected alternative: a `re.search(r"/WX(?!-)")` over the raw text. It closes
the inversion and nothing else — the OBJCXX-only occurrence and the comment
still satisfy it, and #640 shows those are the shapes that actually got written.

## Sibling evasions — decided, not silently widened

**`COMPILE_WARNING_AS_ERROR OFF` (target property or `CMAKE_` cache variable):
OUT of scope.** CMake uses this property only to decide whether *it* adds a
warnings-as-errors flag; there is no mechanism by which setting it `OFF` removes
a literal flag written into `target_compile_options`. Under the VS generator it
lowers to MSBuild `TreatWarningAsError=false`, whose `/WX-` precedes
`AdditionalOptions` on the CL command line, and MSVC takes the last of the pair
— so our explicit `/WX` still wins. It cannot defeat the repaired policy, and
the converse case fails closed already: a tree that dropped the literal `/WX`
and relied on `COMPILE_WARNING_AS_ERROR ON` instead would go RED on the missing
token, which is the correct direction. Asserting a property *value* would be a
new policy about how the flag is spelled, which #774 does not describe.

**Blanket `/wd####`: OUT of scope.** A per-warning suppression is a different
axis from the two tokens the policy names: it does not invert `/W4` or `/WX`, it
narrows what `/W4` reports. Refusing it wholesale would forbid the legitimate
targeted suppression (the tree has none today, so nothing is being grandfathered
in), and "how many suppressions are too many" is a threshold nobody has decided.
Any future `/wd` arrives on a PR of its own and is a review judgement there.
Recorded here as a known, deliberate non-goal rather than an oversight.

**`/W0` and `/w`: IN scope**, unlike the two above, because they are the
disable spellings of `/W4` itself. Leaving them out would close `/WX-` while
leaving its exact twin one character away, which is the same defect #774
reports rather than a wider one.

## Risks

- **Over-strictness on a legitimate tree.** A CXX-scoped genex, a bare
  `add_compile_options(/W4 /WX)`, and a config-conditional genex must all still
  pass. Pinned by an inverse test on each shape, and by the real tree.
- **Genex parsing.** The stack scan handles the two forms this repository
  writes. A `COMPILE_LANGUAGE` buried inside `$<AND:...>` is handled by
  collecting *every* `COMPILE_LANGUAGE` mention inside a span; a span with no
  mention is conservatively kept, so the failure direction is "counts as
  reaching C/C++", never "silently excluded".
- **Comment stripping.** `#` inside a CMake string would be stripped early. The
  only consequence is a false RED on the warning policy, which is visible, and
  the repository has no such line.

## Tests

`tests/scripts/test_check_windows_portability.py`, all through the existing
`assert_rejected` / `run_checker` harness so they exercise the real checker
process:

1. `/WX-` on the CXX arm is rejected (the #774 headline).
2. The #640 `74ba3823f` shape — `/WX-` on CXX with the only bare `/WX` on
   OBJCXX — is rejected.
3. `/W44996` and `/WXsomething` do not satisfy the policy.
4. `/W4 /WX` present but cancelled by a later `/w` is rejected.
5. Flags present only inside a `#` comment are rejected.
6. Inverse pins: the honest `$<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>` genex, the
   bare `add_compile_options(/W4 /WX)` fixture, and a `$<$<CONFIG:Debug>:...>`
   genex all pass.
7. The real tree's `cmake/CompilerWarnings.cmake` satisfies the repaired policy
   with the OBJCXX arm excluded and no negation present.

Every one of 1-5 and 7 must FAIL against the base checker; that is the
red-before evidence, and it is also what `check-pr-size.py` re-executes for a
`governance_checker` change.

## Gates

- `scripts/agent-preflight.sh`
- `python3 -m pytest tests/scripts/ --ignore=tests/scripts/test_cpu_kernel_bench.py`
- `python3 scripts/check-windows-portability.py` against the real tree (green
  before and after — this repair does not change the verdict on `main`).

No build, no GPU, no benchmark: this is a source-contract checker.

## Evidence

Base checker, run over the real `CMakeLists.txt` plus a substituted
`cmake/CompilerWarnings.cmake`, evaluating the assertion exactly as line 1710
spells it:

```
PASS  BASE (main: honest /W4 /WX on CXX)
PASS  PR#640@74ba3823f (/WX- on CXX; bare /WX only on OBJCXX)
PASS  decoy: /WX- appended on CXX after the real /WX
PASS  decoy: /w (all warnings off) appended on CXX
PASS  substring-only tokens: /W44996 and /WXsomething
```

Four of those five are policy inversions and the gate cannot see any of them.
The per-test red-before and green-after runs are recorded in the PR body.

## Stop conditions

- If closing the language scope turns the real tree RED, stop and report: the
  tree, not the checker, would then be the finding.
- If a shape outside `$<$<COMPILE_LANGUAGE:...>:...>` is needed to decide
  reachability, stop and return `NEEDS_DECISION` rather than growing a CMake
  generator-expression evaluator inside a portability checker.

## Now

`ENG-RELEASE-WINDOWS` does not change lifecycle state. This is a checker repair
inside an already-`ACTIVE` row; no `STATUS`/`BENCHMARKS` projection is owed.
