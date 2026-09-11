ID: ISSUE-GH-482
Title: `check-snapshot-pins` FALSE-POSITIVES on the correct pinned form: a file naming the revision then enumerating shards inside `snapshots/<rev>/` is reported UNPINNED, and the message names a C++ header with no Python equivalent. No live FP only because `online_gate.py:3631` case-shadows (`snapshot` parameter vs `snapshots` marker)
Row: GATE-PIN-UNPINNED-SNAPSHOTS
State: UNKNOWN
Kind: bug
GitHub: 482
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:133`

### Frozen archive evidence

> | [#482](https://github.com/mudler/vllm.cpp/issues/482) | `GATE-PIN-UNPINNED-SNAPSHOTS` | `check-snapshot-pins` FALSE-POSITIVES on the correct pinned form: a file naming the revision then enumerating shards inside `snapshots/<rev>/` is reported UNPINNED, and the message names a C++ header with no Python equivalent. No live FP only because `online_gate.py:3631` case-shadows (`snapshot` parameter vs `snapshots` marker) | bug |

## Resolution

-
