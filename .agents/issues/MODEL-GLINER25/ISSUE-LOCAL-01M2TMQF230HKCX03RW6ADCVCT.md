ID: ISSUE-LOCAL-01M2TMQF230HKCX03RW6ADCVCT
Title: Port GLiNER2.5 (fastino/gliner2.5-multi-v1) as first encoder-only model
Row: MODEL-GLINER25
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

vllm.cpp has no encoder-only (BERT-class) model, no DeBERTa v2 encoder, no disentangled attention, and no GLiNER2 pooler head. The GLiNER2.5 model (fastino/gliner2.5-multi-v1) provides zero-shot NER and structured extraction. Port it with CPU + GPU (CUDA + ROCm) support, OpenAI-compatible serving, and a LocalAI backend.

## Resolution

-
