# Control CUDA graph capture

Use `VLLM_CPP_CUDAGRAPH=0` to disable CUDA graph capture for all decode and
draft graphs. The process reads this variable once, so set it before you start
the application.

```sh
VLLM_CPP_CUDAGRAPH=0 ./build/examples/vllm-cli \
  --model /path/to/model \
  --prompt "The capital of France is" \
  --max-tokens 8
```

With capture disabled, the engine runs the same forward operations eagerly.
CPU, Metal, and Vulkan also use the eager path because those backends do not
support CUDA graph capture.

## Per-model controls

Three decode drivers also have a local A/B control:

| Driver | Environment variable |
|---|---|
| DeepSeek V4 | `VT_V4_DECODE_GRAPH` |
| DFlash draft | `VT_DFLASH_GRAPH` |
| Laguna | `VT_LAGUNA_DECODE_GRAPH` |

Set one of these variables to `0` to disable capture only for that driver.
`VLLM_CPP_CUDAGRAPH=0` overrides all three controls.

## Speculative query lengths

`VT_SPEC_GRAPH_MAX_QLENS` limits the number of speculative query lengths that
each driver captures. The default is `2`: the steady-state length and one
clamped length. Set the value to `0` to remove the limit. A step beyond the
limit runs eagerly.

The limit bounds memory use. Each captured shape retains two ring slots for an
`[S, vocab]` f32 logits block and an `[S, H]` hidden-state block.

Prefill and mixed batches always run eagerly. Most decode drivers capture only
query length 1. The Qwen3.5 drivers can capture more than one speculative query
length.

## Capture failures

Disabling capture and failing to capture are different states. With capture
disabled, the engine computes the output eagerly. If capture starts and then
fails, the engine reports the error instead of returning an uncomputed buffer.

Laguna can use an auxiliary stream during capture. The capture scope joins that
stream before it closes the captured segment. You do not need to configure this
behavior.

See the [CUDA graph break specification](../../.agents/specs/eng-cudagraph-break.md)
for implementation history and test evidence.
