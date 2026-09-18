ID: ISSUE-LOCAL-01M2TJ5PJCV43GH1ZDT2ZTAYKQ
Title: W4: adapter load/config (PEFTHelper, parse_fine_tuned_lora_name, LoRAModel.from_local_checkpoint)
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

Port vLLM's LoRA adapter loading layer to C++: PEFTHelper (adapter_config.json parsing, scaling computation, validation), parse_fine_tuned_lora_name (weight name parsing with optional WeightsMapper), and LoRAModel.from_local_checkpoint (safetensors reading, unexpected module check, tensor-to-LoRALayerWeights assignment). Ports test_peft_helper.py and test_lora_checkpoints.py.

## Resolution

-
