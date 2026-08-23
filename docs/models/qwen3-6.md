# Qwen3.6

Qwen3.6 is a dense and mixture-of-experts family. It runs through the shared
paths, so [the quickstart](../QUICKSTART.md) and [the usage guide](../USAGE.md)
cover starting a server and sending a request.

This page carries what is specific to its quantized checkpoints: which forms of
`lm_head` load, and one merged GEMM that fires only under a condition the
checkpoint controls.

## Which `lm_head` forms load

Publishers do not agree on how weights are stored, and a single repo can change
it between revisions. One 27B repo named NVFP4 silently became FP8 throughout.

The table below is about `lm_head`. The same three forms are accepted for the
attention, MLP, and `linear_attn` projections, in both compressed-tensors
(`weight_packed` plus `weight_global_scale`) and ModelOpt (`weight` plus
`weight_scale_2`) naming. The Qwen3.6 dense family accepts all three forms in
use, so pick a checkpoint by its quality and not by its head.

| `lm_head.weight` | Companion tensors | Seen in |
|---|---|---|
| `BF16` | none | `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` |
| `F8_E4M3` | `lm_head.weight_scale`, per-output-channel or per-tensor | `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` |
| `U8` NVFP4 | `lm_head.weight_scale` plus `weight_scale_2` (ModelOpt), or `weight_global_scale` (compressed-tensors) | `nvidia/Qwen3.6-27B-NVFP4` |

The head is dequantized to BF16 at load, so all three cost the same memory once
running. Any other dtype fails at load with a message naming what it saw.

## The merged `in_proj_qkvz` GEMM

A `modelopt_mixed` checkpoint, meaning `nvidia/Qwen3.6-27B-NVFP4` and the
35B-A3B that shares the tower, keeps its `linear_attn` input projections in FP8
W8A8. Those two per-layer projections are packed into one merged
`in_proj_qkvz` GEMM, mirroring vLLM's `MergedColumnParallelLinear`.

The merge fires only when the two shards carry a bitwise-identical per-tensor
`input_scale`, because one GEMM quantizes the activation once. A checkpoint
whose scales differ keeps the two separate GEMMs automatically, with no action
from you. `VT_GDN_MERGED_QKVZ_FP8=0` restores the two GEMMs in the same binary,
which is the same-binary A/B escape hatch.
