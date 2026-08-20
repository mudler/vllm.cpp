# Stream routed experts from disk

Use expert streaming when the checkpoint is larger than available memory.

A mixture-of-experts checkpoint larger than the box can hold can be run by
keeping the routed-expert weights on disk and paging slices into a bounded
resident cache. It is **off by default** and it is a **capacity** feature, not a
throughput one: it targets single-user and low-concurrency use, and at high
concurrency every step touches most of the experts, so there is nothing left to
save.

```sh
VT_MOE_EXPERT_STREAM=1 \
VT_MOE_EXPERT_STREAM_SLOTS=4000 \
  ./build/examples/vllm-cli --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
                   --prompt "The capital of France is" --max-tokens 16
```
