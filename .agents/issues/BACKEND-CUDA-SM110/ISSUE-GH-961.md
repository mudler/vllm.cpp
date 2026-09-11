ID: ISSUE-GH-961
Title: `tests/tools/test_online_gate_startup.py:259` guards `shellcheck` absence with a check that cannot fire, so an absent instrument reads as a code verdict: `test_serve_low_tools` raises `FileNotFoundError: 'shellcheck'` instead of skipping. Filed by the sm_110 baseline lane ([#955](https://github.com/mudler/vllm.cpp/issues/955)), where the leased `thor:gpu0` worker carries no `shellcheck`, so the baseline names the failure as a known entry rather than an sm_110 fact. Indexed late: the issue was opened 2026-08-15 and its index row was lost with the unmerged repair of PR [#956](https://github.com/mudler/vllm.cpp/pull/956)
Row: BACKEND-CUDA-SM110
State: UNKNOWN
Kind: bug
GitHub: 961
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:642`

### Frozen archive evidence

> | [#961](https://github.com/mudler/vllm.cpp/issues/961) | `BACKEND-CUDA-SM110` | `tests/tools/test_online_gate_startup.py:259` guards `shellcheck` absence with a check that cannot fire, so an absent instrument reads as a code verdict: `test_serve_low_tools` raises `FileNotFoundError: 'shellcheck'` instead of skipping. Filed by the sm_110 baseline lane ([#955](https://github.com/mudler/vllm.cpp/issues/955)), where the leased `thor:gpu0` worker carries no `shellcheck`, so the baseline names the failure as a known entry rather than an sm_110 fact. Indexed late: the issue was opened 2026-08-15 and its index row was lost with the unmerged repair of PR [#956](https://github.com/mudler/vllm.cpp/pull/956) | bug |

## Resolution

-
