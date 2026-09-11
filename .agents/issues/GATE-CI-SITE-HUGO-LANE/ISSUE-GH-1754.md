ID: ISSUE-GH-1754
Title: **`agent-record` is RED on `main` and on every pull request cut from it: `test_rendered_benchmark_index_links_resolve_to_emitted_pages` shells out to `hugo` and no job in `ci.yml` installs it, so `subprocess.run` raises `FileNotFoundError` before any assertion runs and `unittest` reports an ERROR.** Introduced by [#1714](https://github.com/mudler/vllm.cpp/pull/1714) at `1db7e59cf`; the scheduled baseline at `deb53c6a3` was green on this job and `08c81a892` and every scheduled run after it is red. FIXED IN FLOW by installing the renderer on the lane -- the same `peaceiris/actions-hugo@v3` action, the same `extended: true`, and the same `0.146.3` pin the `gh-pages` job publishes on -- plus `hugo version` as the step's first line and `tests/scripts/test_ci_site_lane.py` holding the two pins equal and the setup step present and ordered before the render. A SKIP GUARD WAS REJECTED AS THE FIX and is deliberately not in this change: it retires the red by arranging for the assertion to run nowhere, which [PR #1726](https://github.com/mudler/vllm.cpp/pull/1726) demonstrates -- its `agent-record` is SUCCESS having rendered no site. The guard is still wanted for a developer box without Hugo and is separately owned by [#1722](https://github.com/mudler/vllm.cpp/issues/1722) / #1726, which edits a disjoint file, so the two compose. [#1764](https://github.com/mudler/vllm.cpp/issues/1764) names this red as one of its three and is not closed by this change
Row: GATE-CI-SITE-HUGO-LANE
State: UNKNOWN
Kind: bug
GitHub: 1754
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:686`

### Frozen archive evidence

> | [#1754](https://github.com/mudler/vllm.cpp/issues/1754) | `GATE-CI-SITE-HUGO-LANE` | **`agent-record` is RED on `main` and on every pull request cut from it: `test_rendered_benchmark_index_links_resolve_to_emitted_pages` shells out to `hugo` and no job in `ci.yml` installs it, so `subprocess.run` raises `FileNotFoundError` before any assertion runs and `unittest` reports an ERROR.** Introduced by [#1714](https://github.com/mudler/vllm.cpp/pull/1714) at `1db7e59cf`; the scheduled baseline at `deb53c6a3` was green on this job and `08c81a892` and every scheduled run after it is red. FIXED IN FLOW by installing the renderer on the lane -- the same `peaceiris/actions-hugo@v3` action, the same `extended: true`, and the same `0.146.3` pin the `gh-pages` job publishes on -- plus `hugo version` as the step's first line and `tests/scripts/test_ci_site_lane.py` holding the two pins equal and the setup step present and ordered before the render. A SKIP GUARD WAS REJECTED AS THE FIX and is deliberately not in this change: it retires the red by arranging for the assertion to run nowhere, which [PR #1726](https://github.com/mudler/vllm.cpp/pull/1726) demonstrates -- its `agent-record` is SUCCESS having rendered no site. The guard is still wanted for a developer box without Hugo and is separately owned by [#1722](https://github.com/mudler/vllm.cpp/issues/1722) / #1726, which edits a disjoint file, so the two compose. [#1764](https://github.com/mudler/vllm.cpp/issues/1764) names this red as one of its three and is not closed by this change | bug |

## Resolution

-
