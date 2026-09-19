ID: ISSUE-LOCAL-01M2C26CSZWB7WVRS5H7YPW4S8
Title: DeepSeek-V4 vision: the device MoE router takes one bias per call, so an image step is refused by name instead of served on CUDA
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

DispRoute (src/vllm/model_executor/models/deepseek_v4.cpp:437) refuses any step carrying image rows on the two DEVICE router arms, because MoeDeviceKernels::route and route_ip each take ONE bias pointer for the whole call and have no per-row selector. Routing an image row on the text bias would be fluent and wrong, so the refusal is correct and deliberate; what is missing is the selector itself. The CPU arm SqrtSoftplusRouteTopk already selects per token between the text bias exp_probs_b and the vision bias exp_probs_b_vl, and on a hash layer an image row leaves the tid2eid route while a text row in the same step keeps it. ForwardDevice builds V4Backend dev_be{device=true}, so EVERY served request on this architecture takes the be.device arm, which makes this refusal the last known blocker to serving an image end to end on CUDA. Measured on thor:gpu0 (sm_110, CUDA 13.0.88): the request travels the whole registered forward and stops here.

## Resolution

-
