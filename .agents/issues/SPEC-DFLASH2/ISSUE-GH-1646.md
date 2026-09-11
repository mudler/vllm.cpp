ID: ISSUE-GH-1646
Title: **`tests/tools/` held 351 cases across 20 suites and NO lane ran one of them.** Measured at `e100e64e1` on a clean worktree: `python3 -m unittest discover -s tests/tools -t . -p "test_*.py"` reports `Ran 351 tests in 21.067s / OK`, standard library only, zero skips, no GPU and no vLLM wheel — and no workflow, no CTest registration and no `scripts/agent-preflight.sh` line executed it. The only `unittest` invocations in `.github/workflows/` are four `tests.scripts.*` modules; preflight's `SUITES` loop runs `tests/scripts/$suite.py` only; a tree-wide grep for `tests.tools` outside the directory returns prose in `.agents/` and four unrelated path constants in `scripts/check-snapshot-pins.py`. Worse than untested: the suites are QUOTED AS EVIDENCE — `.agents/parity-ledger.md` carries "all tools 34/34" on five `SERVE-GATE-ONLINE` rows and `.agents/upstream-sync.md:38` records "34 of the 233 `tests/tools` cases" — so they read as gating in every document that cites them while being reachable only by an agent who typed the command. What they cover is not marginal: `test_oracle_pin.py` is the [#520](https://github.com/mudler/vllm.cpp/issues/520) oracle-identity assertion, `test_gpu_clock_state.py` is the [#543](https://github.com/mudler/vllm.cpp/issues/543) clock attribution every ratio rests on. FIXED IN FLOW by one preflight line and one CI step, DISCOVERED rather than enumerated because an enumerated list is a shared file every new suite must edit, which is the record-lock shape `AGENTS.md` §Records forbids. Found while wiring [#1562](https://github.com/mudler/vllm.cpp/issues/1562)'s refusal gate, which would otherwise have landed dead; not owned by that row's subject matter
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1646
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:593`

### Frozen archive evidence

> | [#1646](https://github.com/mudler/vllm.cpp/issues/1646) | `SPEC-DFLASH2` | **`tests/tools/` held 351 cases across 20 suites and NO lane ran one of them.** Measured at `e100e64e1` on a clean worktree: `python3 -m unittest discover -s tests/tools -t . -p "test_*.py"` reports `Ran 351 tests in 21.067s / OK`, standard library only, zero skips, no GPU and no vLLM wheel — and no workflow, no CTest registration and no `scripts/agent-preflight.sh` line executed it. The only `unittest` invocations in `.github/workflows/` are four `tests.scripts.*` modules; preflight's `SUITES` loop runs `tests/scripts/$suite.py` only; a tree-wide grep for `tests.tools` outside the directory returns prose in `.agents/` and four unrelated path constants in `scripts/check-snapshot-pins.py`. Worse than untested: the suites are QUOTED AS EVIDENCE — `.agents/parity-ledger.md` carries "all tools 34/34" on five `SERVE-GATE-ONLINE` rows and `.agents/upstream-sync.md:38` records "34 of the 233 `tests/tools` cases" — so they read as gating in every document that cites them while being reachable only by an agent who typed the command. What they cover is not marginal: `test_oracle_pin.py` is the [#520](https://github.com/mudler/vllm.cpp/issues/520) oracle-identity assertion, `test_gpu_clock_state.py` is the [#543](https://github.com/mudler/vllm.cpp/issues/543) clock attribution every ratio rests on. FIXED IN FLOW by one preflight line and one CI step, DISCOVERED rather than enumerated because an enumerated list is a shared file every new suite must edit, which is the record-lock shape `AGENTS.md` §Records forbids. Found while wiring [#1562](https://github.com/mudler/vllm.cpp/issues/1562)'s refusal gate, which would otherwise have landed dead; not owned by that row's subject matter | bug |

## Resolution

-
