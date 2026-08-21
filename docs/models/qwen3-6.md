# Qwen3.6

Use this page for Qwen3.6 checkpoints, commands, supported arms, and current limitations.

## Quantized checkpoints: which `lm_head` forms load

Publishers do not agree on how weights are stored, and a single repo can change
it between revisions (one 27B "NVFP4" repo silently became FP8 throughout).
The table below is about `lm_head`; the same three forms are accepted for the
attention, MLP and `linear_attn` projections, in both compressed-tensors
(`weight_packed` + `weight_global_scale`) and ModelOpt (`weight` +
`weight_scale_2`) naming. For the Qwen3.6 dense family we accept all three
forms in use, so pick a checkpoint by its quality, not by its head:

| `lm_head.weight` | Companion tensors | Seen in |
|---|---|---|
| `BF16` | none | `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` |
| `F8_E4M3` | `lm_head.weight_scale` (per-output-channel or per-tensor) | `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` |
| `U8` NVFP4 | `lm_head.weight_scale` + `weight_scale_2` (ModelOpt) or `weight_global_scale` (compressed-tensors) | `nvidia/Qwen3.6-27B-NVFP4` |

The head is dequantized to BF16 at load, so all three cost the same memory once
running. Any other dtype fails at load with a message naming what it saw.

A `modelopt_mixed` checkpoint (`nvidia/Qwen3.6-27B-NVFP4`, and the 35B-A3B that
shares the tower) keeps its `linear_attn` input projections in FP8 W8A8, and
those two per-layer projections are packed into ONE merged `in_proj_qkvz` GEMM,
mirroring vLLM's `MergedColumnParallelLinear`. The merge only fires when the two
shards carry a bitwise-identical per-tensor `input_scale`, since one GEMM
quantizes the activation once; a checkpoint whose scales differ keeps the two
separate GEMMs automatically. `VT_GDN_MERGED_QKVZ_FP8=0` restores the two GEMMs
in the same binary.
