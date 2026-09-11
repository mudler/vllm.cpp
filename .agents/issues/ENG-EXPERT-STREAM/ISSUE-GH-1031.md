ID: ISSUE-GH-1031
Title: CLOSED as a duplicate of [#1022](https://github.com/mudler/vllm.cpp/issues/1022), and this row is the corrected one rather than the filed one. As filed it said `check-agent-record` and `test_check_agent_record` are RED on `origin/main` because `.agents/issue-index.md` lists issue #995 twice, and that the repair needed a contract decision plus a checker-semantics spec. It did not: [#1022](https://github.com/mudler/vllm.cpp/issues/1022) had already read the two #995 rows and found neither WELL-FORMED, and `ff264cb82` (PR [#1025](https://github.com/mudler/vllm.cpp/pull/1025)) landed the repair on `main` before this branch merged it. Measured at this branch's head rather than inferred from the merge: `python3 scripts/check-agent-record.py` prints `agent record OK: ENGINE=156 MODEL=377 QUANT=82 KERNEL=51 BACKEND=83` and exits 0. CORRECTED IN PLACE, and that is a narrow exception argued here rather than a licence: the row had not landed, this branch added it, and the net diff against `origin/main` is still additions only — which `scripts/check-issue-index-append-only.py --base origin/main` is what checks. Once it lands `merge=union` makes it permanent and un-correctable, so leaving a filed-and-refuted claim in the record was the more expensive option
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1031
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:302`

### Frozen archive evidence

> | [#1031](https://github.com/mudler/vllm.cpp/issues/1031) | `ENG-EXPERT-STREAM` | CLOSED as a duplicate of [#1022](https://github.com/mudler/vllm.cpp/issues/1022), and this row is the corrected one rather than the filed one. As filed it said `check-agent-record` and `test_check_agent_record` are RED on `origin/main` because `.agents/issue-index.md` lists issue #995 twice, and that the repair needed a contract decision plus a checker-semantics spec. It did not: [#1022](https://github.com/mudler/vllm.cpp/issues/1022) had already read the two #995 rows and found neither WELL-FORMED, and `ff264cb82` (PR [#1025](https://github.com/mudler/vllm.cpp/pull/1025)) landed the repair on `main` before this branch merged it. Measured at this branch's head rather than inferred from the merge: `python3 scripts/check-agent-record.py` prints `agent record OK: ENGINE=156 MODEL=377 QUANT=82 KERNEL=51 BACKEND=83` and exits 0. CORRECTED IN PLACE, and that is a narrow exception argued here rather than a licence: the row had not landed, this branch added it, and the net diff against `origin/main` is still additions only — which `scripts/check-issue-index-append-only.py --base origin/main` is what checks. Once it lands `merge=union` makes it permanent and un-correctable, so leaving a filed-and-refuted claim in the record was the more expensive option | bug |

## Resolution

-
