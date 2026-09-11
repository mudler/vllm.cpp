ID: ISSUE-GH-1373
Title: `.agents/benchmark-record.md:3` self-declares "Append-only forensic record" but `.gitattributes` gives `merge=union` only to `.agents/issue-index.md` (`.gitattributes:7`), and the file is 7 lines long so nothing else supplies one. Two append-only record surfaces, both declared append-only in prose, and the union driver on one of them. AGENTS.md §Records admits "a genuinely append-only file that can union-merge" as a record shape and warns that "if N concurrent pull requests edit file F, that file is a lock"; without the attribute this file IS that lock, because two branches that each append a section collide on the same tail with zero overlapping content. MEASURED on [#1349](https://github.com/mudler/vllm.cpp/pull/1349) in one afternoon: merging `9e1a5e573` (#1369) gave `CONFLICT (content): Merge conflict in .agents/benchmark-record.md`, and merging `2f67c9358` gave the same conflict again as the ONLY conflict once `docs/BENCHMARKS.md` had settled — both pure tail appends on both sides (49 lines ours, 393 and 234 theirs), each costing a full resolve/re-verify/re-gate cycle, the second losing a merge race. SCOPE for the fixing row, so it does not over-promise: per [#883](https://github.com/mudler/vllm.cpp/issues/883) the union driver is LOCAL, so GitHub will still report these appends as conflicting in its own UI; what it buys is that `git merge` and `git merge-tree` stop conflicting locally, which is where the cycles are spent. Also owed there, not here: whether `scripts/roll-benchmark-record.py` can produce a NON-tail edit, since union-merge is only sound while every write is genuinely an append. NOT fixed in flow — changing a merge driver changes how every future concurrent append resolves, which is a semantic change to the repository's conflict behaviour and wants its own spec and fresh review rather than a one-line ride-along in a multimodal PR
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: bug
GitHub: 1373
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:462`

### Frozen archive evidence

> | [#1373](https://github.com/mudler/vllm.cpp/issues/1373) | `ENG-RECORD-CONFLICT-SURFACES` | `.agents/benchmark-record.md:3` self-declares "Append-only forensic record" but `.gitattributes` gives `merge=union` only to `.agents/issue-index.md` (`.gitattributes:7`), and the file is 7 lines long so nothing else supplies one. Two append-only record surfaces, both declared append-only in prose, and the union driver on one of them. AGENTS.md §Records admits "a genuinely append-only file that can union-merge" as a record shape and warns that "if N concurrent pull requests edit file F, that file is a lock"; without the attribute this file IS that lock, because two branches that each append a section collide on the same tail with zero overlapping content. MEASURED on [#1349](https://github.com/mudler/vllm.cpp/pull/1349) in one afternoon: merging `9e1a5e573` (#1369) gave `CONFLICT (content): Merge conflict in .agents/benchmark-record.md`, and merging `2f67c9358` gave the same conflict again as the ONLY conflict once `docs/BENCHMARKS.md` had settled — both pure tail appends on both sides (49 lines ours, 393 and 234 theirs), each costing a full resolve/re-verify/re-gate cycle, the second losing a merge race. SCOPE for the fixing row, so it does not over-promise: per [#883](https://github.com/mudler/vllm.cpp/issues/883) the union driver is LOCAL, so GitHub will still report these appends as conflicting in its own UI; what it buys is that `git merge` and `git merge-tree` stop conflicting locally, which is where the cycles are spent. Also owed there, not here: whether `scripts/roll-benchmark-record.py` can produce a NON-tail edit, since union-merge is only sound while every write is genuinely an append. NOT fixed in flow — changing a merge driver changes how every future concurrent append resolves, which is a semantic change to the repository's conflict behaviour and wants its own spec and fresh review rather than a one-line ride-along in a multimodal PR | bug |

## Resolution

-
