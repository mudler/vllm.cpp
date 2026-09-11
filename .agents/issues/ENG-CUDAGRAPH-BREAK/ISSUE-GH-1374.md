ID: ISSUE-GH-1374
Title: W6, the last stage: the graph-eligibility predicate moves out of the two model files that each re-derived it and into `GPUModelRunner::execute_model`, which names the step's ACTUAL uniform query length once through `v1::GraphEligibleQueryLen` and ships it on `ModelForwardInput::uniform_query_len`. That closes [#1020](https://github.com/mudler/vllm.cpp/issues/1020) together with a `(S, q, spec)` ring key, which was a LIVE collision rather than the enabler #1020 called it. **What did NOT move is 'except at the break points'**: no driver in this tree serves a prefill or a mixed batch under any predicate, so that needs a prefill capture driver nobody has written and whose benefit the spec's D5 already refutes on this hardware — recorded as a publishable negative in `## Owed` rather than as a stage that ran out of window
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: feature
GitHub: 1374
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:463`

### Frozen archive evidence

> | [#1374](https://github.com/mudler/vllm.cpp/issues/1374) | `ENG-CUDAGRAPH-BREAK` | W6, the last stage: the graph-eligibility predicate moves out of the two model files that each re-derived it and into `GPUModelRunner::execute_model`, which names the step's ACTUAL uniform query length once through `v1::GraphEligibleQueryLen` and ships it on `ModelForwardInput::uniform_query_len`. That closes [#1020](https://github.com/mudler/vllm.cpp/issues/1020) together with a `(S, q, spec)` ring key, which was a LIVE collision rather than the enabler #1020 called it. **What did NOT move is 'except at the break points'**: no driver in this tree serves a prefill or a mixed batch under any predicate, so that needs a prefill capture driver nobody has written and whose benefit the spec's D5 already refutes on this hardware — recorded as a publishable negative in `## Owed` rather than as a stage that ran out of window | feature |

## Resolution

-
