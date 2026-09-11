ID: ISSUE-GH-933
Title: Measure gateability of the `llama-cpp-unsloth` oracle by BUILDING it and RUNNING `Qwen3.8-2.4T-A95B UD-Q1_0` on it. The oracle is pinned at `36fe8e1cc` (branch `iq1-narrow`) and records `gateable = no`, because the IQ1_XXXS port is grounded in the fork's SOURCE, read and cited, which is weaker than a running comparison. It is the only place ggml type 66 is defined: the vllm.cpp pin `237ad9b96` ends at `Q1_0 = 41` and `ggml-org` master `ad1de39e0` at `Q2_0 = 42`, while type 66 carries 96.92 % of that checkpoint's parameters. Running it needs the full 370 GiB checkpoint and, per Unsloth's documentation, at least 450 GB of RAM. Until then the ported arm has no running oracle, which is what `gateable = no` makes visible
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: task
GitHub: 933
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:262`

### Frozen archive evidence

> | [#933](https://github.com/mudler/vllm.cpp/issues/933) | `ENG-EXPERT-STREAM` | Measure gateability of the `llama-cpp-unsloth` oracle by BUILDING it and RUNNING `Qwen3.8-2.4T-A95B UD-Q1_0` on it. The oracle is pinned at `36fe8e1cc` (branch `iq1-narrow`) and records `gateable = no`, because the IQ1_XXXS port is grounded in the fork's SOURCE, read and cited, which is weaker than a running comparison. It is the only place ggml type 66 is defined: the vllm.cpp pin `237ad9b96` ends at `Q1_0 = 41` and `ggml-org` master `ad1de39e0` at `Q2_0 = 42`, while type 66 carries 96.92 % of that checkpoint's parameters. Running it needs the full 370 GiB checkpoint and, per Unsloth's documentation, at least 450 GB of RAM. Until then the ported arm has no running oracle, which is what `gateable = no` makes visible | task |

## Resolution

-
