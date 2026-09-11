ID: ISSUE-GH-797
Title: Inference-time CPU weight offload is unported while vLLM ships the whole surface at the pin: `OffloadConfig` with a three-value backend enum, the UVA arm (`cpu_offload_gb` + dotted-segment `cpu_offload_params` targeting, pinned host copy, zero-copy device view) and the layer-group `PrefetchOffloader`. A pure MIRROR FLOOR — no design freedom, no secondary oracle — and the dense half of #149, whose CPU-MoE half is `ENG-HYBRID-PLACEMENT`. Two constraints shape it: the non-UVA fallback is incompatible with CUDA graphs (upstream's own test appends `--enforce-eager` whenever UVA is off) and `should_pin_memory()` carries a unified-memory caveat that makes the feature inert on GB10, so its memory and speed gates need a discrete-GPU rig we do not have. The matrix's recorded `gpu_model_runner.py:445,913` anchors were STALE and are re-derived to `:939`
Row: ENG-WEIGHT-OFFLOAD
State: UNKNOWN
Kind: feature
GitHub: 797
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:199`

### Frozen archive evidence

> | [#797](https://github.com/mudler/vllm.cpp/issues/797) | `ENG-WEIGHT-OFFLOAD` | Inference-time CPU weight offload is unported while vLLM ships the whole surface at the pin: `OffloadConfig` with a three-value backend enum, the UVA arm (`cpu_offload_gb` + dotted-segment `cpu_offload_params` targeting, pinned host copy, zero-copy device view) and the layer-group `PrefetchOffloader`. A pure MIRROR FLOOR — no design freedom, no secondary oracle — and the dense half of #149, whose CPU-MoE half is `ENG-HYBRID-PLACEMENT`. Two constraints shape it: the non-UVA fallback is incompatible with CUDA graphs (upstream's own test appends `--enforce-eager` whenever UVA is off) and `should_pin_memory()` carries a unified-memory caveat that makes the feature inert on GB10, so its memory and speed gates need a discrete-GPU rig we do not have. The matrix's recorded `gpu_model_runner.py:445,913` anchors were STALE and are re-derived to `:939` | feature |

## Resolution

-
