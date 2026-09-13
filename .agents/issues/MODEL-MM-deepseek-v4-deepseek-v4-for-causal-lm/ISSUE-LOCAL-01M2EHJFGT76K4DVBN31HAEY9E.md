ID: ISSUE-LOCAL-01M2EHJFGT76K4DVBN31HAEY9E
Title: DeepSeek-V4 routes --device cpu into ForwardDevice: the forward arm is keyed on gather_logits, not on the queue's device
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

MEASURED 2026-09-13 (rc job b622dd45-d763-41d6-9fe3-c1822100163d, dgx:gpu0, Release, base 7a62a7fca, the 82,438,622,112-byte DeepSeek-V4-Flash-Vision-Exp UD-IQ1_S GGUF). vllm-cli was given --device cpu. The engine constructed, auto-fit the KV cache and then died with: 'vt: DeepseekV4 DEVICE forward (W7-device) not implemented' at src/vllm/model_executor/models/deepseek_v4.cpp:4658, which is VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending) inside DeepseekV4Model::ForwardDevice. A CPU run reached the DEVICE forward. WHY: ForwardDeepseekV4ForCausalLM selects the arm on input.gather_logits ALONE (src/vllm/model_executor/models/deepseek_v4_registry.cpp:253) and consults nothing about input.queue.device. The runner sets gather_logits from 'gather = LogitsGatherEnabled() && step.prompt_logprob_indices.empty()' (src/vllm/v1/worker/gpu/runner.cpp:3074), which is true on every default step on every device, so the host composition at deepseek_v4_registry.cpp:294 is unreachable unless VT_LOGITS_GATHER is turned off or a request asks for prompt logprobs. The refusal message itself names the host composition (DeepseekV4Model::Forward / DeepseekV4ForwardHost) as where the CPU path lands, so the route contradicts the message. NOT FIXED HERE: this was found while repairing the NDEBUG-dependent verdict of test_deepseek_v4_multigroup_kv, and changing the forward routing is a production change that needs its own spec and fresh review.

## Resolution

-
