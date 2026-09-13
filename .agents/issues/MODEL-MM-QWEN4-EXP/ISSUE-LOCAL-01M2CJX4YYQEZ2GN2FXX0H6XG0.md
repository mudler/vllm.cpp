ID: ISSUE-LOCAL-01M2CJX4YYQEZ2GN2FXX0H6XG0
Title: run-doctest-selected.sh guards zero SELECTION but not zero ASSERTIONS
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

`scripts/run-doctest-selected.sh` refuses a filter that selects no test case, and that is the only blindness it removes. A filtered run whose selected cases all EARLY-RETURN still reads green, and every `CUDA W7:` case in `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp` has exactly that shape.

DEMONSTRATED by the W7 fresh reviewer on a CUDA-less host. A doctest binary whose cases early-return through a `SkipNoCuda`-shaped guard reports `--count` 2, so the guard prints `selected=2` and execs. The run then reports `test cases: 2 | 2 passed`, **`assertions: 0`**, and exits 0. The guard saw a non-zero selection and was satisfied; nothing asserted anything. That is the same class of false green the guard was written for (`0 passed | 0 failed | 18 skipped`, exit 0), reached by a different route: selection is non-zero, execution is empty.

WHAT IS OWED. A `--min-assertions <N>` companion guard: parse doctest's `assertions:` summary line from the real run and refuse when the count is below the caller's floor. A mutation arm then has to state how many assertions it expects to execute, and a host that silently skipped the device cases cannot pass.

SCOPE NOTE, so the priority is honest. The existing guard is currently OPT-IN. Nothing in CI and no gate invokes `run-doctest-selected.sh`; its only reference in the tree is a comment at `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp:1665` telling a reviewer to run a filtered mutation arm through it. So this is a gap in an instrument a human chooses to reach for, not in a gate that runs. Wiring the guard into something that runs is the other half of the same question and belongs with this fix.

## Resolution

-
