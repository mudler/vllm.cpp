ID: ISSUE-GH-1405
Title: `scripts/check-gate-commands.py` exists to answer whether a gated row's `## Gates` names a command that CAN FAIL, and it drops `true`, `echo ok` and anything containing a pipe for that reason. It does not drop `git diff`. Measured at `96ed8346f`: `runnable_commands` over `.agents/specs/eng-cudagraph-break.md` returns exactly `['git diff']`, and that one always-zero command is the whole reason `ENG-CUDAGRAPH-BREAK` sits in `RUNNABLE_BASELINE` (`:414`). Its `## Gates` names no test binary, least of all `tests/test_qwen3_5_decode_graph_seam`, whose W6 case exited 139 on `main` ([#1390](https://github.com/mudler/vllm.cpp/issues/1390), [#1394](https://github.com/mudler/vllm.cpp/issues/1394)). The EXECUTION gates were surveyed at the same revision and are sound — `ctest` fails on exit 139 (`tests/CMakeLists.txt:30`, `.github/workflows/ci.yml:1013`), `scripts/mutation-harness.py:99-113` reads the exit code and `Status: FAILURE!` and says the summary line is not the authority, `tools/bench/gdn_packed_component.py:1893-1902` requires `Status: SUCCESS!` beside its pinned totals, and `scripts/music3-vocoder-conv-ab.sh:88` recovers `${PIPESTATUS[0]}` past its grep — so NO execution gate reads a crashed run as green, and the exposure is the record layer plus [#1376](https://github.com/mudler/vllm.cpp/issues/1376). Sharper than it reads: a SIGSEGV TRUNCATES the assertion count (135 crashed vs 138 complete on one unchanged file), so a gate pinning a total reports a crash as count drift. NOT fixed in flow: adding `git diff` to `_CANNOT_FAIL` moves rows out of the runnable population, reds `--check` and the exact `RUNNABLE_BASELINE` pin in `tests/scripts/test_check_gate_commands.py`, and `AGENTS.md` routes a semantic checker change to its own row, spec and fresh review
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1405
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:472`

### Frozen archive evidence

> | [#1405](https://github.com/mudler/vllm.cpp/issues/1405) | `ENG-CUDAGRAPH-BREAK` | `scripts/check-gate-commands.py` exists to answer whether a gated row's `## Gates` names a command that CAN FAIL, and it drops `true`, `echo ok` and anything containing a pipe for that reason. It does not drop `git diff`. Measured at `96ed8346f`: `runnable_commands` over `.agents/specs/eng-cudagraph-break.md` returns exactly `['git diff']`, and that one always-zero command is the whole reason `ENG-CUDAGRAPH-BREAK` sits in `RUNNABLE_BASELINE` (`:414`). Its `## Gates` names no test binary, least of all `tests/test_qwen3_5_decode_graph_seam`, whose W6 case exited 139 on `main` ([#1390](https://github.com/mudler/vllm.cpp/issues/1390), [#1394](https://github.com/mudler/vllm.cpp/issues/1394)). The EXECUTION gates were surveyed at the same revision and are sound — `ctest` fails on exit 139 (`tests/CMakeLists.txt:30`, `.github/workflows/ci.yml:1013`), `scripts/mutation-harness.py:99-113` reads the exit code and `Status: FAILURE!` and says the summary line is not the authority, `tools/bench/gdn_packed_component.py:1893-1902` requires `Status: SUCCESS!` beside its pinned totals, and `scripts/music3-vocoder-conv-ab.sh:88` recovers `${PIPESTATUS[0]}` past its grep — so NO execution gate reads a crashed run as green, and the exposure is the record layer plus [#1376](https://github.com/mudler/vllm.cpp/issues/1376). Sharper than it reads: a SIGSEGV TRUNCATES the assertion count (135 crashed vs 138 complete on one unchanged file), so a gate pinning a total reports a crash as count drift. NOT fixed in flow: adding `git diff` to `_CANNOT_FAIL` moves rows out of the runnable population, reds `--check` and the exact `RUNNABLE_BASELINE` pin in `tests/scripts/test_check_gate_commands.py`, and `AGENTS.md` routes a semantic checker change to its own row, spec and fresh review | bug |

## Resolution

-
