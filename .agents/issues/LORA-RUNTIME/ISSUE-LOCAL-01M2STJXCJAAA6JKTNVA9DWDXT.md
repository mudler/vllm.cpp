ID: ISSUE-LOCAL-01M2STJXCJAAA6JKTNVA9DWDXT
Title: LoRA W3: metadata + mapping (LoRAMapping, convert_mapping, compute_meta, sgmv segmentation, PunicaWrapperBase state)
Row: LORA-RUNTIME
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

W3 of lora-adapter.md: port vLLM's LoRA metadata and mapping layer — LoRAMapping, convert_mapping (punica_wrapper/utils.py:54-160), compute_meta (utils.py:15-41), sgmv prefill segmentation, and PunicaWrapperBase state machine (punica_base.py:124-299). W1 (CPU punica brick) and W2 (packed + wrapped layers) are landed; the wrapped layers take index arrays as parameters but nothing produces them yet. W3 builds the LoRAMapping → index-tensor pipeline that connects the scheduler batch to the punica apply. Ports test_punica_ops.py sgmv segment cases + test_layers_utils.py.

## Resolution

-
