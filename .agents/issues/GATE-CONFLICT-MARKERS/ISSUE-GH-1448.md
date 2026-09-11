ID: ISSUE-GH-1448
Title: **`scripts/check-pr-size.py`'s `classify_path` has no entry for a per-run `docs/bench-evidence/<run-id>/<file>` directory, so its own suite is red on `main`.** `BENCH_EVIDENCE` matches exactly one path segment, and the ten files of `docs/bench-evidence/gdn-replayssm-w0-20260818/` (landed 2026-08-18) match none of it. Measured at `9ecaf1bb3`: sweeping every tracked path through `classify_path` leaves exactly 10 unclassified, and `test_every_tracked_and_current_change_path_is_classified` is RED in a detached worktree at that SHA. That suite is in no preflight `SUITES` entry, so preflight is green over it; it surfaces only through the checker-evidence contract, where it reads as `ERROR: HEAD checker/test pair failed for 'scripts/check-pr-size.py'` and looks like a defect in the change under review. Registering any NEW checker requires editing `CREATION_MUTATIONS`, so every future checker was blocked behind it. Fourth instance after [#856](https://github.com/mudler/vllm.cpp/issues/856), [#668](https://github.com/mudler/vllm.cpp/issues/668) and [#989](https://github.com/mudler/vllm.cpp/issues/989). FIXED in flow by a `BENCH_EVIDENCE_RUN` pattern restricted to the extensions the directory carries, excluding `.md` and `.json` because the `evidence` arm is tested before `public_document` and would silently reclassify `docs/bench-evidence/mxfp4-qwen/*`. Verified to move exactly those 10 paths and no others
Row: GATE-CONFLICT-MARKERS
State: UNKNOWN
Kind: bug
GitHub: 1448
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:501`

### Frozen archive evidence

> | [#1448](https://github.com/mudler/vllm.cpp/issues/1448) | `GATE-CONFLICT-MARKERS` | **`scripts/check-pr-size.py`'s `classify_path` has no entry for a per-run `docs/bench-evidence/<run-id>/<file>` directory, so its own suite is red on `main`.** `BENCH_EVIDENCE` matches exactly one path segment, and the ten files of `docs/bench-evidence/gdn-replayssm-w0-20260818/` (landed 2026-08-18) match none of it. Measured at `9ecaf1bb3`: sweeping every tracked path through `classify_path` leaves exactly 10 unclassified, and `test_every_tracked_and_current_change_path_is_classified` is RED in a detached worktree at that SHA. That suite is in no preflight `SUITES` entry, so preflight is green over it; it surfaces only through the checker-evidence contract, where it reads as `ERROR: HEAD checker/test pair failed for 'scripts/check-pr-size.py'` and looks like a defect in the change under review. Registering any NEW checker requires editing `CREATION_MUTATIONS`, so every future checker was blocked behind it. Fourth instance after [#856](https://github.com/mudler/vllm.cpp/issues/856), [#668](https://github.com/mudler/vllm.cpp/issues/668) and [#989](https://github.com/mudler/vllm.cpp/issues/989). FIXED in flow by a `BENCH_EVIDENCE_RUN` pattern restricted to the extensions the directory carries, excluding `.md` and `.json` because the `evidence` arm is tested before `public_document` and would silently reclassify `docs/bench-evidence/mxfp4-qwen/*`. Verified to move exactly those 10 paths and no others | bug |

## Resolution

-
