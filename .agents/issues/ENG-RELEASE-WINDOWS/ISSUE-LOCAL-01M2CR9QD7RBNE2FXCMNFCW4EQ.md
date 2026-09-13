ID: ISSUE-LOCAL-01M2CR9QD7RBNE2FXCMNFCW4EQ
Title: test_model_loader_gguf expected string missing DeepseekV41ForCausalLM
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

Commit 4a9d33792 added DeepseekV41ForCausalLM to the registry but did not update the hardcoded expected architecture list in test_model_loader_gguf.cpp:225

## Resolution

fixed by adding DeepseekV41ForCausalLM to the expected string in test_model_loader_gguf.cpp
