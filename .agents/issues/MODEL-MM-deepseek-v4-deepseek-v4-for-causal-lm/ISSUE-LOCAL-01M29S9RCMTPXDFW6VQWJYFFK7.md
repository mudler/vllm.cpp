ID: ISSUE-LOCAL-01M29S9RCMTPXDFW6VQWJYFFK7
Title: DeepSeek-V4 vision: the EXL3 arm's official-vision accounting block is gated by nothing
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

src/vllm/model_executor/models/deepseek_v4_weights.cpp, inside LoadDeepseekV4Exl3, guards a block on DeepSeekV4ShardsCarryVision(shards) that requires each DeepSeekV4OfficialVisionExpectedTensors name, inserts it into routed and increments accounted. Deleting the block entirely builds clean and leaves BOTH loader suites fully green: test_deepseek_v4_mm_loader 16/16 and test_deepseek_v4_exl3_loader 22/22. Its dense twin in the same file IS gated and reds exactly one case when deleted. Nothing drives the EXL3 copy because every committed vision fixture uses OfficialVisionOptions, which sets quant_method fp8 and dense_routed_experts true -- the released vehicle's dense shape -- so no case ever reaches an exl3 quant_method checkpoint that also carries the vision group. Found by fresh review 2026-09-12.

## Resolution

2026-09-12: fixed in this change. tests/vllm/models/test_deepseek_v4_mm_loader.cpp gains Exl3VisionOptions() -- TwoLayerHashOptions unchanged, so quant_method stays exl3 and dense_routed_experts stays false and BuildVisionFixture writes the four EXL3 rank shards alongside the 27-tensor vision group -- and the case 'dsv4 vision safetensors: the EXL3 arm ROUTES and ACCOUNTS FOR the official vision group', which loads that fixture through the production entry LoadDeepseekV4ForCausalLMWeights, REQUIREs has_exl3_weights so it cannot pass against the dense twin, and checks accounted_tensors equals the vision-free EXL3 load plus the group size, with the group size independently pinned at 27. PROVEN to catch the defect by mutation: deleting the accounting block from LoadDeepseekV4Exl3 built clean (MUT_BUILD_RC=0) and CHANGED the binary (md5 7767ff922a67fa6e772069367758ce3f -> e11a9bc84dd001341db8ca14094ec044), and the new case then RED with 'test_deepseek_v4_mm_loader.cpp:636: FATAL ERROR: REQUIRE( msg.empty() ) is NOT correct!' at 'test cases: 1 | 0 passed | 1 failed', which is the EXL3 arm's totality pass refusing the unrouted vision tensor by name. Restored byte-for-byte (source sha256 back to dbd4257a9cefe2e6e686dbd72c21cfcc605a9e022b7d4df9bd32204a0ef7d5c8, binary md5 back to 7767ff92 after a forced rebuild) and both suites green: test_deepseek_v4_mm_loader 17/17 with 1155 assertions, test_deepseek_v4_exl3_loader 22/22 with 613.
