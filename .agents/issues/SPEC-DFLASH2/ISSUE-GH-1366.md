ID: ISSUE-GH-1366
Title: DFlash causality diverged from upstream on the `use_swa` arm: the legacy fallback read the RESOLVED `is_sliding`, which `dflash_config.use_swa` forces true, while upstream reads the RAW `layer_types` (`bool(layer_types) and layer_types[i] == "sliding_attention"`, `qwen3_dflash.py:58-67` @ vllm#52816 head `19c93519`). Upstream's own `_resolve_layer_attention` docstring table states `layer_types=None` + `use_swa=True` -> causal False and names `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` as that shape, so such a DFlash1 draft ran EVERY layer causal here and non-causal upstream. Acceptance-only and token-gate-invisible, because the verify is lossless. Uncaught because upstream's parametrize table has no `use_swa` row either, so the ported cases were faithful to the ported table and silent about the arm. Second divergence in the same resolution: `is_causal` was honoured only when `.is_boolean()`, while upstream tests presence and coerces (`if is_causal is not None: return bool(is_causal)`), so `"is_causal": 0` fell through in silence -- and the GGUF arm was already more permissive through `KvI64`, so the two containers disagreed with each other as well as with upstream. Found by the fresh reviewer of `SPEC-DFLASH2` W1 and fixed IN FLOW on `row/SPEC-DFLASH2-W1`, red-first: the two docstring rows redden 4 assertions before the repair
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1366
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:459`

### Frozen archive evidence

> | [#1366](https://github.com/mudler/vllm.cpp/issues/1366) | `SPEC-DFLASH2` | DFlash causality diverged from upstream on the `use_swa` arm: the legacy fallback read the RESOLVED `is_sliding`, which `dflash_config.use_swa` forces true, while upstream reads the RAW `layer_types` (`bool(layer_types) and layer_types[i] == "sliding_attention"`, `qwen3_dflash.py:58-67` @ vllm#52816 head `19c93519`). Upstream's own `_resolve_layer_attention` docstring table states `layer_types=None` + `use_swa=True` -> causal False and names `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` as that shape, so such a DFlash1 draft ran EVERY layer causal here and non-causal upstream. Acceptance-only and token-gate-invisible, because the verify is lossless. Uncaught because upstream's parametrize table has no `use_swa` row either, so the ported cases were faithful to the ported table and silent about the arm. Second divergence in the same resolution: `is_causal` was honoured only when `.is_boolean()`, while upstream tests presence and coerces (`if is_causal is not None: return bool(is_causal)`), so `"is_causal": 0` fell through in silence -- and the GGUF arm was already more permissive through `KvI64`, so the two containers disagreed with each other as well as with upstream. Found by the fresh reviewer of `SPEC-DFLASH2` W1 and fixed IN FLOW on `row/SPEC-DFLASH2-W1`, red-first: the two docstring rows redden 4 assertions before the repair | bug |

## Resolution

-
