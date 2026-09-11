ID: ISSUE-GH-1058
Title: `main` is red on `check-commit-trailers` at `e34d71379e70`: the #1054 squash body carried no trailer block, so the landed commit has no `FOLLOWING_AGENTS_PROTOCOL` paragraph and neither the `Following-Agents-Protocol` nor the `AI-Assisted` trailer. Not the `---------` shape #861 closed, and not repairable, because `main` is squash-only and history is not rewritable. The row already owns the missing enforcement in [#870](https://github.com/mudler/vllm.cpp/issues/870), the CI `--filled` pull-request-body guard, and this is the live instance it is owed for
Row: GATE-SQUASH-TRAILERS
State: UNKNOWN
Kind: bug
GitHub: 1058
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:296`

### Frozen archive evidence

> | [#1058](https://github.com/mudler/vllm.cpp/issues/1058) | `GATE-SQUASH-TRAILERS` | `main` is red on `check-commit-trailers` at `e34d71379e70`: the #1054 squash body carried no trailer block, so the landed commit has no `FOLLOWING_AGENTS_PROTOCOL` paragraph and neither the `Following-Agents-Protocol` nor the `AI-Assisted` trailer. Not the `---------` shape #861 closed, and not repairable, because `main` is squash-only and history is not rewritable. The row already owns the missing enforcement in [#870](https://github.com/mudler/vllm.cpp/issues/870), the CI `--filled` pull-request-body guard, and this is the live instance it is owed for | bug |

## Resolution

-
