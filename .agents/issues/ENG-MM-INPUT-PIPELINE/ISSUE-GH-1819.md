ID: ISSUE-GH-1819
Title: **`scripts/mm/tower_skip_rss.sh` configured both build directories with `-DVLLM_CPP_BUILD_EXAMPLES=OFF` and then ran `ninja -C "$d" -j 4 vllm-server`, and `vllm-server` is an `examples/` target** -- the `OUTPUT_NAME` of `server` (`examples/CMakeLists.txt:91,108`), in a directory the root `CMakeLists.txt:2828` adds only under `if(VLLM_CPP_BUILD_EXAMPLES)`. Reproduced with the harness's own flags: the configure returns 0 and `ninja` answers `unknown target 'vllm-server'`, so the run `exit 4`s at arm A before any RSS exists. The block landed on `main` in `bacb71109` (#1364) and had never been executed. **Nothing could catch it**: `tests/scripts/test_tower_skip_rss_report.py` covers `--report-only`, `--check-source` and `--stage-check` -- every path needing no checkpoint -- while the configure, the build, `run_arm`, the `/health` poll and the kill/wait only ever run on a leased box, so the suite was 41/41 green over a harness that could not build its own binary. FIXED IN FLOW: `-DVLLM_CPP_BUILD_EXAMPLES=ON` (measured: `ninja -j 4 vllm-server` then returns 0 and writes `<build>/examples/vllm-server`, the one file of that name in the tree and the path `docs/USAGE.md:54,95,128,204` names), the binary is NAMED rather than found by a `find` piped into `head -1`, and a new `--dry-run` prints the `cmake`/`ninja`/`run_arm` invocations out of the same variables the run issues them from and asserts that CMake defines the requested target under those flags -- statically, so it builds nothing and runs in CI, plus a live `ninja -t targets` prong on an already-configured tree that skips BY NAME when there is none. `DryRunTests` runs it against the script as committed, which is the case that reds on this defect, and against scratch copies with the flag flipped OFF, the flag dropped, and the binary path pointed away from where CMake writes it
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 1819
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:678`

### Frozen archive evidence

> | [#1819](https://github.com/mudler/vllm.cpp/issues/1819) | `ENG-MM-INPUT-PIPELINE` | **`scripts/mm/tower_skip_rss.sh` configured both build directories with `-DVLLM_CPP_BUILD_EXAMPLES=OFF` and then ran `ninja -C "$d" -j 4 vllm-server`, and `vllm-server` is an `examples/` target** -- the `OUTPUT_NAME` of `server` (`examples/CMakeLists.txt:91,108`), in a directory the root `CMakeLists.txt:2828` adds only under `if(VLLM_CPP_BUILD_EXAMPLES)`. Reproduced with the harness's own flags: the configure returns 0 and `ninja` answers `unknown target 'vllm-server'`, so the run `exit 4`s at arm A before any RSS exists. The block landed on `main` in `bacb71109` (#1364) and had never been executed. **Nothing could catch it**: `tests/scripts/test_tower_skip_rss_report.py` covers `--report-only`, `--check-source` and `--stage-check` -- every path needing no checkpoint -- while the configure, the build, `run_arm`, the `/health` poll and the kill/wait only ever run on a leased box, so the suite was 41/41 green over a harness that could not build its own binary. FIXED IN FLOW: `-DVLLM_CPP_BUILD_EXAMPLES=ON` (measured: `ninja -j 4 vllm-server` then returns 0 and writes `<build>/examples/vllm-server`, the one file of that name in the tree and the path `docs/USAGE.md:54,95,128,204` names), the binary is NAMED rather than found by a `find` piped into `head -1`, and a new `--dry-run` prints the `cmake`/`ninja`/`run_arm` invocations out of the same variables the run issues them from and asserts that CMake defines the requested target under those flags -- statically, so it builds nothing and runs in CI, plus a live `ninja -t targets` prong on an already-configured tree that skips BY NAME when there is none. `DryRunTests` runs it against the script as committed, which is the case that reds on this defect, and against scratch copies with the flag flipped OFF, the flag dropped, and the binary path pointed away from where CMake writes it | bug |

## Resolution

-
