# DSpark

Use this page for DSpark checkpoints, commands, supported arms, and current limitations.

## DSpark drafts: the exact checkpoints

A DSpark draft is a SEPARATE checkpoint named by the `model` key of
`--speculative-config`. A repo id alone is not a pin, because a checkpoint can be
re-quantized in place under an unchanged name, so the revision is part of the
identity.

| Draft | Repo and revision | File | Bytes | sha256 |
|---|---|---|---|---|
| Qwen3.8-27B, 5 layers against a 64-layer target | `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | `model.safetensors` | 2 718 576 122 | `9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786` |

That draft declares `architectures: ["DSparkDraftModel"]` with `model_type:
"qwen3"` and `block_size: 7`, which is the pair
[vllm#52197](https://github.com/vllm-project/vllm/pull/52197) routes to the Qwen3
DSpark lane and which this engine mirrors ahead of its pinned oracle
(`SPEC-DSPARK-QWEN3-ROUTING`,
[#1193](https://github.com/mudler/vllm.cpp/issues/1193)). **It has not been run
here yet**: the token-exact gate against the pinned oracle needs the 2.53 GiB
download and GPU time, and both are pending developer authority, so the routing
is gated on CPU and the decode is not.

The two layouts that already run are the native
`deepseek-ai/dspark_qwen3_*_block7` drafts and the Speculators-format
`RedHatAI/*.dspark` drafts; the DeepSeek-V4 DSpark draft, whose weights ship
inside the DeepSeek-V4 target, is refused by name.

**Which refusal you actually get today.** Point the server or the C API at a
DeepSeek-V4 DSpark draft and the message is the named DeepSeek-V4 refusal, the
one the classification produces. `LoadedEngine::FromModelDir` resolves a
`dspark` speculative config ONCE, at the top of the function
([#1225](https://github.com/mudler/vllm.cpp/issues/1225)), before it opens the
target directory and long before it loads the draft, so the classification is now
the FIRST thing a DSpark run meets. An earlier writing of this paragraph said the
draft loader's "the draft config must carry target_layer_ids and mask_token_id"
won instead; that was true while the draft load ran ahead of the resolution, and
it stopped being true when the resolution was hoisted.

Two messages still come out in front of it, and both are the resolution's own.
A `dspark` run that names no `num_speculative_tokens` against a draft whose
config carries no `n_predict` is refused for the missing `k` first, because that
check sits ahead of the classification in the same branch. And a `.gguf` target
takes the GGUF branch above the hoist, which carries its own named refusal for a
GGUF DSpark target (`SPEC-DSPARK`). Either way the draft is refused and nothing
loads it as a Qwen3 draft.
