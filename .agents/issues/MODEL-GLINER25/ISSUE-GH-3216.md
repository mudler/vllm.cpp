ID: ISSUE-GH-3216
Title: Port GLiNER2.5 (fastino/gliner2.5-multi-v1) zero-shot NER to vllm.cpp
Row: MODEL-GLINER25
State: OPEN
Kind: UNKNOWN
GitHub: 3216
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Port the GLiNER2.5 model (DeBERTa v2 encoder + disentangled attention + GLiNER2 pooler head) into vllm.cpp as the first encoder-only (BERT-class) model. vLLM has no DeBERTa support; the vllm-factory plugin (ddickmann/vllm-factory) is the reference oracle.
>
> Scope: DeBERTa v2 encoder, GLiNER2 pooler head (SpanRep, CountLSTM, classifier), model registration, OpenAI-compatible serving, CPU + GPU (CUDA + ROCm) support, LocalAI backend.
>
> Local issue: ISSUE-LOCAL-01M2TMQF230HKCX03RW6ADCVCT
> Row: MODEL-GLINER25
> Spec: .agents/specs/gliner2.5.md

## Resolution

-
