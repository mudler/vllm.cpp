# Oracle: `unslothai/llama.cpp` (branch `iq1-narrow`)

A FORK of `ggml-org/llama.cpp`, admitted for one narrow reason: it is the only
place the sub-IQ1_S quant encodings are defined, and a published checkpoint this
project must run uses one of them. It does not replace the `llama-cpp` oracle,
and it never outranks vLLM.

## Why a fork is admitted here

`unsloth/Qwen3.8-2.4T-A95B-GGUF UD-Q1_0` stores its routed experts as ggml type
**66**, which is about 97 % of that model's parameters. That type is in no
upstream llama.cpp:

| Tree | Revision | Highest type |
|---|---|---|
| vllm.cpp pin | `237ad9b96` | `GGML_TYPE_Q1_0 = 41` |
| `ggml-org/llama.cpp` master, 2026-08-15 | `ad1de39e0` | `GGML_TYPE_Q2_0 = 42`, `COUNT = 43` |
| this fork, branch `iq1-narrow` | `36fe8e1cc` | `GGML_TYPE_IQ1_XXXS = 66`, `COUNT = 67` |

The fork declares three encodings below IQ1_S, all added by the branch head
commit "IQ1_XS, IQ1_XXS, IQ1_XXXS: three quant types below IQ1_S" (2026-08-07):

```
GGML_TYPE_IQ1_XS   = 64, // 1.4375 bpw, 1024-entry grid
GGML_TYPE_IQ1_XXS  = 65, // 1.3125 bpw,  512-entry grid
GGML_TYPE_IQ1_XXXS = 66, // 1.1875 bpw,  256-entry grid
```

The identification is not taken on trust. The bits per weight were derived
independently from the checkpoint's own GGUF offset deltas, before this fork was
located, and came out at **1.1875**, matching the declaration exactly. The block
layout confirms it a second way: `block_iq1_xxxs` is
`{ ggml_half d; uint8_t qs[QK_K/8]; uint8_t sc[QK_K/64]; }`, so 2 + 32 + 4 = 38
bytes per 256 elements, which is 1.1875 bpw.

## Scope, and what this oracle may not do

Use it ONLY for the sub-IQ1_S encodings it defines: block layout, codebook grid,
scale and delta decode, and the reference `vec_dot` and `dequantize_row`. For
everything else, including CPU and GGUF k-quant speed and memory floors, the
oracle is [`llama-cpp`](llama-cpp.md) at its own pin.

It is a mirror source for ONE encoding family, not a design reference. Where
upstream llama.cpp or vLLM defines behavior, that behavior wins, exactly as
`AGENTS.md` requires.

## Gateability

`gateable = no`, and issue #933 owes the measurement.

`AGENTS.md` admits an oracle as gateable only once it demonstrably BUILDS and
RUNS the model. Neither has been done here: the encoding was ported from the
fork's SOURCE, read and cited, which is a different and weaker thing than a
running comparison. Running it needs the full 370 GiB `UD-Q1_0` checkpoint and,
per Unsloth's own documentation, at least 450 GB of RAM.

So the ported arm has no running oracle to compare against yet. That is visible
debt rather than a gap to be discovered later, which is what `gateable = no` is
for.

What HAS been done, so the gap is not overstated: the ported decoder was run on
real `UD-Q1_0` bytes and compared against an independent transcription of the
fork's own `dequantize_row_iq1_xxxs`, bit-identical over 1179648 weights across
three tensor roles, two layers and two shards. See the census section of
[`.agents/specs/expert-streaming.md`](../specs/expert-streaming.md). That removes
transcription error from our side. It cannot detect a defect in the FORK itself,
because both sides read the same source, which is exactly the residue #933 owes.

## Pin

```oracle-pin
id = llama-cpp-unsloth
role = secondary
scope = the sub-IQ1_S quant encodings IQ1_XS (64), IQ1_XXS (65) and IQ1_XXXS (66), which no upstream llama.cpp defines
upstream = https://github.com/unslothai/llama.cpp
pin = 36fe8e1cc7f2b3b8c92fdda0ab07600141921786
pin_label = iq1-narrow
pinned_on = 2026-08-15
gateable = no
evidence = #933
```

## Anchors used by the port

Read at the pin above. Re-verify before relying on them, because a fork branch
can be rebased under a name.

| Piece | Anchor |
|---|---|
| type ids 64, 65, 66 | `ggml/include/ggml.h` enum `ggml_type` |
| `block_iq1_xxxs`, 38 bytes | `ggml/src/ggml-common.h:478-483` |
| `NGRID_IQ1XXXS = 256` | `ggml/src/ggml-common.h:1181` |
| `iq1_xxxs_grid`, uint64, 256 | `ggml/src/ggml-common.h:2095` |
| `iq1_xxxs_grid_gpu`, uint32, 256 | `ggml/src/ggml-common.h:2620` |
| `ggml_vec_dot_iq1_xxxs_q8_K_generic` | `ggml/src/ggml-cpu/quants.c:1281` |
| `dequantize_row_iq1_xxxs` | `ggml/src/ggml-quants.c:2727` |

The delta constant is upstream's own `IQ1S_DELTA` (0.125), reused unchanged by
the fork rather than redefined.
