ID: ISSUE-GH-647
Title: Oracle policy had no fallback and no pin concept: five upstreams beyond vLLM are already compared against (vLLM-Omni, SGLang, llama.cpp, `transformers`, tt-forge) with their pins scattered across individual specs or absent entirely. AGENTS.md now admits a named secondary oracle where vLLM implements nothing, `.agents/oracles/<id>.md` pins each one file-per-oracle, and `check-oracle-pins.py` enforces both directions. The gateability debts for `sglang`, `diffusers` and `tt-forge` stay open on this issue
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: feature
GitHub: 647
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:193`

### Frozen archive evidence

> | [#647](https://github.com/mudler/vllm.cpp/issues/647) | — | Oracle policy had no fallback and no pin concept: five upstreams beyond vLLM are already compared against (vLLM-Omni, SGLang, llama.cpp, `transformers`, tt-forge) with their pins scattered across individual specs or absent entirely. AGENTS.md now admits a named secondary oracle where vLLM implements nothing, `.agents/oracles/<id>.md` pins each one file-per-oracle, and `check-oracle-pins.py` enforces both directions. The gateability debts for `sglang`, `diffusers` and `tt-forge` stay open on this issue | feature |

## Resolution

-
