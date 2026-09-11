ID: ISSUE-GH-1791
Title: **`scripts/dspark-paired-e2e.sh`'s `settle()` can never break early, so a wait for the GPU to drain always spends its full 360 s however fast the box actually drains.** The same idiom as [#1734](https://github.com/mudler/vllm.cpp/issues/1734), found by sweeping `scripts/` for it: `grep -c .` with an `|| echo 0` fallback makes `$n` the two-line string `0\n0`, so `[ "$n" -eq 0 ] && break` answers `integer expression expected` and returns 2 instead of deciding. The BUSY half of the guard works -- a positive count exits 0 and the fallback does not fire -- so only the FREE half is dead, and the failure is in the safe direction, which is why it was paid in silence. This is the FIFTH diagnosis of the idiom in this tree: `scripts/cpu-x86-llamacpp-floor.sh` already carries the removal and the reason in a comment, and a comment in one file is not reachable from another. FIXED IN FLOW with #1734: `|| true` keeps grep's own `0` and swallows only its status. The recurrence gate is `TheIdiomIsGoneFromEveryShellScript` in `tests/scripts/test_ltx25_ab_memwatch.py`, which sweeps every `scripts/*.sh` for a counting `grep`/`pgrep` paired with an `|| echo` fallback outside a comment; run against `27d8bfa70` it names all three live instances, this one included. It is a TRIPWIRE and says so: it reads text, and a `wc -l` with the same fallback walks past it
Row: SPEC-DSPARK
State: UNKNOWN
Kind: bug
GitHub: 1791
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:680`

### Frozen archive evidence

> | [#1791](https://github.com/mudler/vllm.cpp/issues/1791) | `SPEC-DSPARK` | **`scripts/dspark-paired-e2e.sh`'s `settle()` can never break early, so a wait for the GPU to drain always spends its full 360 s however fast the box actually drains.** The same idiom as [#1734](https://github.com/mudler/vllm.cpp/issues/1734), found by sweeping `scripts/` for it: `grep -c .` with an `\|\| echo 0` fallback makes `$n` the two-line string `0\n0`, so `[ "$n" -eq 0 ] && break` answers `integer expression expected` and returns 2 instead of deciding. The BUSY half of the guard works -- a positive count exits 0 and the fallback does not fire -- so only the FREE half is dead, and the failure is in the safe direction, which is why it was paid in silence. This is the FIFTH diagnosis of the idiom in this tree: `scripts/cpu-x86-llamacpp-floor.sh` already carries the removal and the reason in a comment, and a comment in one file is not reachable from another. FIXED IN FLOW with #1734: `\|\| true` keeps grep's own `0` and swallows only its status. The recurrence gate is `TheIdiomIsGoneFromEveryShellScript` in `tests/scripts/test_ltx25_ab_memwatch.py`, which sweeps every `scripts/*.sh` for a counting `grep`/`pgrep` paired with an `\|\| echo` fallback outside a comment; run against `27d8bfa70` it names all three live instances, this one included. It is a TRIPWIRE and says so: it reads text, and a `wc -l` with the same fallback walks past it | bug |

## Resolution

-
