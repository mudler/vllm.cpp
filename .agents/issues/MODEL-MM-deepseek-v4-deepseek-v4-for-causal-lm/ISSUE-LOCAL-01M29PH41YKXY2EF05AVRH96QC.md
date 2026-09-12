ID: ISSUE-LOCAL-01M29PH41YKXY2EF05AVRH96QC
Title: DeepSeek-V4 vision: the spec carries five stale file:line anchors and wrongly records the per-block staging miss as lease-only
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

Second-round review of 4ccee2f40 found two defects in the row's records. (1) Five file:line citations in .agents/specs/deepseek-v4-flash-vision.md point at the wrong lines: deepseek_v4.cpp:1319 twice (the windowed dev_attn guard moved to :1325 when 4ccee2f40 added six comment lines above it), deepseek_v4.cpp:4345 (the kDevicePending refusal is at :4415), model_registry.h:299-356 (MultiModalForwardInput is at :397-403) and inputs.h:20-92 (the four members it names span :20-136). (2) A ## Owed entry claimed the per-block staging miss in DeepSeekV4Vision::Impl::EnsureResident could only be caught on a leased device. That is false: EnsureResident keys on queue.device != weights.device and stages through the backend the tower was constructed with, neither of which is a CUDA predicate, so a hand-built non-CPU vt::Queue plus a host-memory fake vt::Backend runs the staging loop on a CPU host and the staged copies can be counted. A third entry said no committed test sets VT_V4_DEVICE_ATTN, which is misleading because tools/parity/dsv4v_w7_cuda.sh:207 does, though it is not ctest-registered.

## Resolution

2026-09-12: fixed in this change. (1) All five stale anchors corrected in .agents/specs/deepseek-v4-flash-vision.md: deepseek_v4.cpp:1319 -> :1325 at two sites, deepseek_v4.cpp:4345 -> :4415, model_registry.h:299-356 -> :397-403, inputs.h:20-92 -> :20-136. api_server.cpp:373 and test_deepseek_v4_exl3_forward.cpp:443,446 were verified ACCURATE and left alone; the :2958 and mm_reach citations sit inside verbatim historical logs and were deliberately not renumbered. (2) The false lease-only claim is withdrawn and replaced by a committed CPU gate: tests/vllm/models/test_deepseek_v4_vision.cpp 'DeepSeek-V4 vision stages every per-block weight to the queue's device' registers a host-memory fake backend, hand-builds a kXPU vt::Queue and counts staged allocations and copies through DeepSeekV4Vision::Forward -- 15 at depth 1, 23 at depth 2, slope 8. PROVEN to catch the defect by mutation: making EnsureResident skip blocks with index > 0 (MUT_BUILD_RC=0, binary md5 cc44ef7c -> df9c8239) reds it with CHECK(15 == 23) and slope CHECK(0 == 8), test cases: 1 | 0 passed | 1 failed; restoring byte-for-byte (sha256 back to c614f817, md5 back to cc44ef7c after forcing recompile) returns 1 passed and the full suite to 16/16, 7420 assertions. (3) The VT_V4_DEVICE_ATTN entry now names tools/parity/dsv4v_w7_cuda.sh:207 as the committed harness that drives the refusal, while keeping that it is not ctest-registered and needs a manual lease run.
