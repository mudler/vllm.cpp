ID: ISSUE-GH-989
Title: `scripts/check-pr-size.py`'s `classify_path` has no entry for `.agents/reachability.md` (added by `POLICY-NOTHING-LANDS-DEAD`, [#888](https://github.com/mudler/vllm.cpp/issues/888) @ `8f49ac3be`), and it FAILS CLOSED, so `pr-size` — a REQUIRED check — refuses every pull request that touches that guide, and `tests/scripts/test_check_pr_size.py` has been red on `main` ever since. Red SILENTLY: that suite is wired into no CI job and is not in `agent-preflight.sh`'s `SUITES`, so the only thing that ever loads it is `check-pr-size`'s own executable-evidence contract, which fires only when a PR edits a checker — the red is reachable exclusively by the next person who must touch that file, and presents to them as their own breakage (the [#584](https://github.com/mudler/vllm.cpp/issues/584)/[#965](https://github.com/mudler/vllm.cpp/issues/965) shape). Third instance of the class after [#856](https://github.com/mudler/vllm.cpp/issues/856) (`issue-index.md` + the style guides) and [#668](https://github.com/mudler/vllm.cpp/issues/668) (`.agents/oracles/*`), both fixed in flow by the row that tripped over them. FIXED IN FLOW while landing [#960](https://github.com/mudler/vllm.cpp/issues/960), which could not register its new checker's creation mutation without touching `check-pr-size.py` at all. NOT fixed: wiring that suite into CI, which is its own change and would red `main` until this landed
Row: VT-FP8-QUANT-ARCH-GATE
State: UNKNOWN
Kind: bug
GitHub: 989
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:309`

### Frozen archive evidence

> | [#989](https://github.com/mudler/vllm.cpp/issues/989) | `VT-FP8-QUANT-ARCH-GATE` | `scripts/check-pr-size.py`'s `classify_path` has no entry for `.agents/reachability.md` (added by `POLICY-NOTHING-LANDS-DEAD`, [#888](https://github.com/mudler/vllm.cpp/issues/888) @ `8f49ac3be`), and it FAILS CLOSED, so `pr-size` — a REQUIRED check — refuses every pull request that touches that guide, and `tests/scripts/test_check_pr_size.py` has been red on `main` ever since. Red SILENTLY: that suite is wired into no CI job and is not in `agent-preflight.sh`'s `SUITES`, so the only thing that ever loads it is `check-pr-size`'s own executable-evidence contract, which fires only when a PR edits a checker — the red is reachable exclusively by the next person who must touch that file, and presents to them as their own breakage (the [#584](https://github.com/mudler/vllm.cpp/issues/584)/[#965](https://github.com/mudler/vllm.cpp/issues/965) shape). Third instance of the class after [#856](https://github.com/mudler/vllm.cpp/issues/856) (`issue-index.md` + the style guides) and [#668](https://github.com/mudler/vllm.cpp/issues/668) (`.agents/oracles/*`), both fixed in flow by the row that tripped over them. FIXED IN FLOW while landing [#960](https://github.com/mudler/vllm.cpp/issues/960), which could not register its new checker's creation mutation without touching `check-pr-size.py` at all. NOT fixed: wiring that suite into CI, which is its own change and would red `main` until this landed | bug |

## Resolution

-
