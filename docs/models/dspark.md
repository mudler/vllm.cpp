# DSpark

DSpark is a speculative decoding method. A DSpark draft is a **separate
checkpoint**, named by the `model` key of `--speculative-config`, not a head
inside the target.

Read [the speculative decoding guide](../SPECULATIVE-DECODING.md) for the shared
flags and how drafting works. This page carries which DSpark draft layouts load,
which are refused, and which refusal you actually meet first.

## Which draft layouts run

| Layout | State |
|---|---|
| Native `deepseek-ai/dspark_qwen3_*_block7` | Runs |
| Speculators format, `RedHatAI/*.dspark` | Runs |
| `RadixArk/Qwen3.8-27B-DSpark` | Routing is gated on CPU. The decode is not gated |
| The DeepSeek-V4 DSpark draft, whose weights ship inside the DeepSeek-V4 target | **Refused by name** |

## The Qwen3.8 27B draft

A repo id alone is not a pin, because a checkpoint can be re-quantized in place
under an unchanged name, so the revision is part of the identity.

| Field | Value |
|---|---|
| Draft | Qwen3.8-27B, 5 layers against a 64-layer target |
| Repo and revision | `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` |
| File | `model.safetensors` |
| Bytes | 2 718 576 122 |
| sha256 | `9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786` |

That draft declares `architectures: ["DSparkDraftModel"]` with `model_type:
"qwen3"` and `block_size: 7`. That pair is what
[vllm#52197](https://github.com/vllm-project/vllm/pull/52197) routes to the Qwen3
DSpark lane, and this engine mirrors it ahead of its pinned oracle
(`SPEC-DSPARK-QWEN3-ROUTING`,
[#1193](https://github.com/mudler/vllm.cpp/issues/1193)).

**It has not been run here yet.** The token-exact gate against the pinned oracle
needs the 2.53 GiB download and GPU time, and both are pending developer
authority. The routing is gated on CPU. The decode is not.

## Which refusal you meet first

Point the server or the C API at a DeepSeek-V4 DSpark draft and the message is
the named DeepSeek-V4 refusal, the one the classification produces.

`LoadedEngine::FromModelDir` resolves a `dspark` speculative config **once**, at
the top of the function
([#1225](https://github.com/mudler/vllm.cpp/issues/1225)), before it opens the
target directory and long before it loads the draft. The classification is
therefore the first thing a DSpark run meets. Before that hoist the draft
loader's own refusal won instead, which is why an older reading of this page
named a different message.

Two messages still come out in front of it, and both belong to the resolution
itself:

- A `dspark` run that names no `num_speculative_tokens`, against a draft whose
  config carries no `n_predict`, is refused for the missing `k` first. That check
  sits ahead of the classification in the same branch.
- A `.gguf` target takes the GGUF branch above the hoist, which carries its own
  named refusal for a GGUF DSpark target (`SPEC-DSPARK`).

Either way the draft is refused, and nothing loads it as a Qwen3 draft.
