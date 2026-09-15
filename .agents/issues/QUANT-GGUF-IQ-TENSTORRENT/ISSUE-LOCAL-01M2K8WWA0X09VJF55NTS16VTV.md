ID: ISSUE-LOCAL-01M2K8WWA0X09VJF55NTS16VTV
Title: APEX-I-Nano e2e: layernorm program L1 overflow on the P150 after the IQ waves
Row: QUANT-GGUF-IQ-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-15
Updated: 2026-09-15
Closed: -

## Problem

Re-running the APEX-I-Nano e2e on current main (post waves 1-3, build of 5f83792a7 lineage) dies at the first forward: 'Statically allocated dataflow buffers on core range [0-0 - 10-9] grow to 2213200 B which is beyond max L1 size of 1572864 B' (dataflow_buffer.cpp:2608, via MakeMeshWorkloadFromSpecs during Qwen3_5DenseModel::ForwardDevice), in a layernorm/layernorm_large_tensor program (bench settings identical to the 2026-09-14 run: 2 prompts, input 128, output 32, concurrency 2). The 2026-09-14 APEX run (pre-waves tree, commit 6cff72e41 lineage) passed the same layernorm programs and died later in a DRAM OOM inside the keep-quant ttnn::where repair chain, so the L1 overflow is either a regression introduced by waves 1-3 (or #3197/#3199), or an artifact-shape interaction those changes exposed. Evidence: /tmp/row-tt-iq-evidence/apex-e2e-waves.log (new, L1) and apex-e2e.log (old, DRAM OOM at ttnn::where). The APEX e2e owed item in the row spec cannot be adjudicated until this failure is attributed and fixed.

## Resolution

-
