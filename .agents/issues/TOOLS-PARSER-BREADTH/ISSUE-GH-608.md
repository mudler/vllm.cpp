ID: ISSUE-GH-608
Title: **W0 (record backfill) landed 2026-08-13; row `INVENTORIED` → `PARTIAL`. W1 then shipped `inkling`, taking the registry to 42 names / 38 families.** Re-derived from the two REGISTRIES rather than from recipe usage: **five** `--tool-call-parser` names are upstream-only at the pin — `openai`, `inkling`, `minimax_m3` (W1, recipe demand) and `cohere_command3`, `cohere_command4` (W2, ZERO recipe uses, so usage-driven audits miss them). **Only `inkling` was PORTABLE from vLLM source, and it LANDED** (a `ParserEngineToolAdapter` over the already-ported Inkling engine; the gap was the registry face, not the grammar). Of the four left: `minimax_m3` is backed by the Rust crate; `openai` is a stub delegating to `vllm/parser/harmony.py`, which IS vLLM source but wraps the out-of-tree `openai_harmony` package (the SGLang secondary-oracle check was run and REFUSED — vLLM implements this path, so the rule does not admit a secondary); both Cohere names are shims over the out-of-tree `cohere_melody` package. So W1-remaining/W2 each owe a recorded decision before code rather than a text port. `nemotron_json`, `kimi_k3` and `ling3` are in NEITHER registry and arrive with the pin advance, not here. W3 ports upstream's shared `ToolParserTestConfig` harness. The earlier "six missing" framing was usage-derived: it listed `nemotron_json` as portable (it is not registered at the pin) and missed both Cohere entries
Row: TOOLS-PARSER-BREADTH
State: UNKNOWN
Kind: feature
GitHub: 608
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:192`

### Frozen archive evidence

> | [#608](https://github.com/mudler/vllm.cpp/issues/608) | `TOOLS-PARSER-BREADTH` | **W0 (record backfill) landed 2026-08-13; row `INVENTORIED` → `PARTIAL`. W1 then shipped `inkling`, taking the registry to 42 names / 38 families.** Re-derived from the two REGISTRIES rather than from recipe usage: **five** `--tool-call-parser` names are upstream-only at the pin — `openai`, `inkling`, `minimax_m3` (W1, recipe demand) and `cohere_command3`, `cohere_command4` (W2, ZERO recipe uses, so usage-driven audits miss them). **Only `inkling` was PORTABLE from vLLM source, and it LANDED** (a `ParserEngineToolAdapter` over the already-ported Inkling engine; the gap was the registry face, not the grammar). Of the four left: `minimax_m3` is backed by the Rust crate; `openai` is a stub delegating to `vllm/parser/harmony.py`, which IS vLLM source but wraps the out-of-tree `openai_harmony` package (the SGLang secondary-oracle check was run and REFUSED — vLLM implements this path, so the rule does not admit a secondary); both Cohere names are shims over the out-of-tree `cohere_melody` package. So W1-remaining/W2 each owe a recorded decision before code rather than a text port. `nemotron_json`, `kimi_k3` and `ling3` are in NEITHER registry and arrive with the pin advance, not here. W3 ports upstream's shared `ToolParserTestConfig` harness. The earlier "six missing" framing was usage-derived: it listed `nemotron_json` as portable (it is not registered at the pin) and missed both Cohere entries | feature |

## Resolution

-
