# Native MSVC `M_PI` repair — the LTX-2.5 pi spellings

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#720](https://github.com/mudler/vllm.cpp/issues/720)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Predecessor repair:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md), whose Design
already ruled: "Use standard C++20 math constants rather than enabling
non-standard `M_PI`." This repair applies that ruling to the LTX-2.5 sources
that landed after it.

Status: `ACTIVE`. Base `a4313641413394e48e6e7a2f05d8e13d541f8be0`.

## Scope

Remove every `M_PI` from `src/` and `tests/` and replace it with
`std::numbers::pi_v<double>`, the spelling this repository already uses at four
existing sites. Delete the two hand-rolled `#ifndef M_PI / #define M_PI` blocks
in the LTX-2.5 VAEs. Change no value, no expression structure, no tolerance, no
golden, and no runtime behavior.

Explicitly excluded: `_USE_MATH_DEFINES` in any form; any new named pi constant
of our own; `src/vllm/multimodal/video_engine.cpp` (issue #664, PR #677); the
second, unrelated failure in the same portability suite (issue #680).

## Observed baseline and root cause

`M_PI` is a POSIX extension, not standard C++. MSVC's `<cmath>` defines it only
when `_USE_MATH_DEFINES` precedes the include, so every unguarded use is a hard
C2065. `windows-msvc-cpu` on PR #677 reported:

```
src\vllm\model_executor\models\ltx2.cpp(72,58): error C2065: 'M_PI': undeclared identifier
src\vllm\model_executor\models\ltx2.cpp(72,15): error C2737: 'kBeta': const object must be initialized
src\vllm\model_executor\models\ltx2.cpp(557,69): error C2065: 'M_PI': undeclared identifier
src\vllm\model_executor\models\ltx2.cpp(572,76): error C2065: 'M_PI': undeclared identifier
```

The intake counted "three LTX-2.5 TUs", from `git grep -l M_PI -- src include`.
**That count is wrong in both directions**, and measurement rather than grep is
what separates them:

- `ltx2_audio_vae.cpp` and `ltx2_video_vae.cpp` carry their own
  `#ifndef M_PI / #define M_PI 3.14159265358979323846` guard, so both compile
  under MSVC today. They are not build breaks. They are two extra spellings of
  pi — the defect [#687](https://github.com/mudler/vllm.cpp/issues/687) names —
  and `ltx2_video_vae.cpp`'s guard has **zero uses**: it is dead.
- `tests/vllm/models/test_vocoder1d.cpp` has two unguarded uses and **is** a
  hard MSVC break. `git grep -- src include` never looked at `tests/`, and the
  CI log never named it because the library build stops at `ltx2.cpp` first.

So the real MSVC break is two TUs, not three, and one of the two was invisible
to both the CI message and the intake grep.

## Design

`std::numbers::pi_v<double>` from `<numbers>`, available at this project's
`CMAKE_CXX_STANDARD 20` (`CMakeLists.txt:35`), and already the spelling at
`mla_attention.cpp:82`, `deepseek_v4.cpp:602`, `minimax_h3_video_vae.cpp:258`,
and `vocoder1d.cpp:35`. One definition — the standard's — reached by an include
rather than by a macro whose correctness depends on preceding every transitive
`<cmath>`. No fifth spelling is introduced.

`_USE_MATH_DEFINES` is rejected: it is order-dependent across transitive
includes, and it re-declares the POSIX extension instead of removing the
dependency on it.

## Tests and RED evidence

No checker changes. `tests/scripts/test_check_windows_portability.py`'s
`test_real_tree_uses_portable_windows_allocation_and_math` **already** asserts
that no file under `src/` or `tests/` contains `\bM_PI\b`, and it is already RED
on `main` for exactly these four files. That committed assertion is the RED, and
it is also why the `tests/` TU is in scope: the existing contract always covered
it. Because that suite runs in no workflow (#646, #680), the assertion had gone
unheld while the LTX-2.5 lane landed.

The MSVC diagnostic itself is additionally reproduced on Linux by compiling the
real TUs with `-U_GNU_SOURCE -D_ISOC99_SOURCE`, which removes glibc's `M_PI`
from `<cmath>` and so recreates MSVC's header condition on g++.

Value identity is asserted, not assumed, three ways: the bit pattern of both
spellings, a token-sequence equivalence proof that the diff restructures no
expression, and a before/after run of the seven LTX-2.5 and vocoder golden
suites.

## Gates

1. RED: `test_real_tree_uses_portable_windows_allocation_and_math` fails at base
   naming all four files; the Linux MSVC-condition probe reproduces the C2065
   sites at matching line and column.
2. GREEN: the same test passes; the suite's case count is unchanged.
3. Before/after golden equality across `test_vocoder1d`, `test_ltx2`,
   `test_ltx2_vae`, `test_ltx2_device`, `test_ltx2_loader`,
   `test_ltx2_text_encoder`, `test_ltx2_pipeline`.
4. Mutation: reverting the constant re-reds the gate; perturbing pi's value reds
   the goldens, proving arm 3 is not a vacuous green.
5. Full `scripts/agent-preflight.sh --staged` and full `ctest`.
6. Native `windows-msvc-cpu` and `windows-msvc-vulkan` on the PR head. Linux
   cannot substitute for the MSVC compiler gate.

## Risks and stop conditions

The risk is a silent value or associativity change hiding behind a mechanical
substitution — a token gate cannot see a constant that is merely slightly wrong.
The token-equivalence proof and the value-perturbation mutation exist for that.
Stop with `NEEDS_DECISION` if any golden moves; a moved value would mean the
constant is not what it is claimed to be.

## Outcome

The base tree's own committed portability test was already RED, naming
`ltx2.cpp`, `ltx2_video_vae.cpp`, `ltx2_audio_vae.cpp` and
`tests/vllm/models/test_vocoder1d.cpp` — four files, not the three the issue
title counts, and 71 cases ran in both arms with failures going 2 to 1. The
surviving failure, `test_real_unsupported_tier_helper_is_structurally_scoped`,
is pre-existing and owned by #680; it was captured failing on the pristine base
before any edit.

The Linux `-U_GNU_SOURCE -D_ISOC99_SOURCE` probe reproduced the hosted
diagnostic at `ltx2.cpp` 72:58, 557:69 and 572:76 — the same line **and column**
as the MSVC log — and additionally at `test_vocoder1d.cpp` 106:21 and 121:21,
which no MSVC log had yet reached. It emitted nothing for the two VAEs,
confirming by measurement that their guards make them compile.

Value identity holds three ways. `M_PI`, `std::numbers::pi_v<double>` and the
hand-rolled `3.14159265358979323846` are the same double, `400921fb54442d18`,
and the three derived quantities (`sqrt(2/pi)` as float, `pi/2` as float and as
double) are bit-equal. A token-sequence comparison over all four files, 41,191
tokens with the two pi spellings unified, reports pure token substitution: the
only differences are the added `<numbers>` includes and the two removed
hand-rolled defines, so no operator order, cast or operand moved. The seven
golden suites report identical counts before and after — 185 cases and 17,378
assertions, all passing in both arms.

That third arm is not vacuous. Perturbing pi to `3.14159` at the three
`ltx2.cpp` sites builds clean and reds 60 assertions in `test_ltx2` and 5 in
`test_ltx2_pipeline`, so those goldens do see this constant. `test_ltx2_vae`
stays green under that mutation — it does not reach these sites, and its
unchanged result is therefore evidence of nothing in particular.

Reverting the constant to `M_PI` in `ltx2.cpp` still **builds clean on Linux**
(exit 0) while re-reding the portability gate and the MSVC-condition probe. That
is the whole shape of this bug: the Linux build is structurally blind to it, and
only a source-pattern gate or the MSVC compiler can see it. Every mutation was
restored from the index and verified by `sha256sum -c`, never by `git status`.
